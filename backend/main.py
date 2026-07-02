import asyncio
import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    from crypto_utils import create_session, decrypt_image, encrypt_image
    from processor import pipeline as processor_pipeline
    from scanner import scan, ScanResult
    from utils import UPLOAD_DIR
except ImportError:
    from .crypto_utils import create_session, decrypt_image, encrypt_image
    from .processor import pipeline as processor_pipeline
    from .scanner import scan, ScanResult
    from .utils import UPLOAD_DIR

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Track processed files per batch so download only includes current batch
_batch_files: dict[str, list[str]] = {}

app = FastAPI(title="SecureVision")

VALID_OPERATIONS = [
    "translate", "rotate", "reflect", "crop", "shear",
    "edge_detect", "equalize", "brightness", "contrast",
    "color", "sharpness", "noise_remove", "grayscale", "hsv",
    "segment", "blur", "detect_objects",
]

_PARAM_RULES = {
    "translate":     {"dx": (int, -2000, 2000), "dy": (int, -2000, 2000)},
    "rotate":        {"angle": (float, -360.0, 360.0)},
    "reflect":       {"axis": (int, -1, 1)},
    "crop":          {"x": (int, 0, None), "y": (int, 0, None), "width": (int, 0, None), "height": (int, 0, None)},
    "shear":         {"shear_factor": (float, -2.0, 2.0)},
    "edge_detect":   {"threshold1": (int, 0, 500), "threshold2": (int, 0, 500)},
    "brightness":    {"factor": (float, 0.1, 10.0)},
    "contrast":      {"factor": (float, 0.1, 10.0)},
    "color":         {"factor": (float, 0.1, 10.0)},
    "sharpness":     {"factor": (float, 0.1, 10.0)},
    "segment":       {"threshold": (float, 0.0, 1.0)},
    "blur":          {"sigma": (float, 0.1, 20.0)},
    "noise_remove":  {"size": (int, 3, 11)},
    "detect_objects": {"conf": (float, 0.05, 0.5)},
}


def _val_err(status: int, **kw):
    raise HTTPException(status_code=status, detail={**kw, "valid_operations": VALID_OPERATIONS})


def validate_operations(ops: list) -> None:
    if not ops:
        _val_err(400, error="No operations specified")
    for i, op_spec in enumerate(ops):
        if isinstance(op_spec, str):
            op_name = op_spec
            op_params = {}
        elif isinstance(op_spec, dict):
            op_name = op_spec.get("operation", "")
            op_params = op_spec.get("params", {})
        else:
            _val_err(422, error=f"Invalid operation entry at index {i}: expected string or dict")
        if op_name not in VALID_OPERATIONS:
            _val_err(422, error=f"Unknown operation: {op_name}")
        rules = _PARAM_RULES.get(op_name, {})
        for key, (ptype, pmin, pmax) in rules.items():
            val = op_params.get(key)
            if val is None:
                continue
            try:
                val = ptype(val)
            except (ValueError, TypeError):
                _val_err(422, error=f"Operation '{op_name}': param '{key}' must be {ptype.__name__}, got {type(val).__name__}")
            if pmin is not None and val < pmin:
                _val_err(422, error=f"Operation '{op_name}': param '{key}'={val} below minimum {pmin}")
            if pmax is not None and val > pmax:
                _val_err(422, error=f"Operation '{op_name}': param '{key}'={val} above maximum {pmax}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# API key auth (optional)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY is None:
        return await call_next(request)
    path = request.url.path
    public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    if path in public_paths:
        return await call_next(request)
    if (FRONTEND_DIR / path.lstrip("/")).is_file():
        return await call_next(request)
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {"status": "ok"}





# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@app.get("/session-key")
@limiter.limit("30/minute")
async def session_key(request: Request):
    return create_session()


# ---------------------------------------------------------------------------
# Upload + scan
# ---------------------------------------------------------------------------
@app.post("/upload")
@limiter.limit("20/minute")
async def upload_endpoint(request: Request, encrypted_image: UploadFile = File(...), session_id: str = Form(...)):
    if not encrypted_image.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    t0 = time.perf_counter()
    try:
        enc_bytes = await encrypted_image.read()
        raw_bytes = decrypt_image(enc_bytes, session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")
    decrypt_ms = round((time.perf_counter() - t0) * 1000, 1)

    ext = Path(encrypted_image.filename).suffix.lower() or ".bin"
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / stored_filename
    file_path.write_bytes(raw_bytes)

    try:
        result: ScanResult = scan(str(file_path))
    except Exception as e:
        os.unlink(str(file_path))
        raise HTTPException(status_code=400, detail=str(e))

    if result.sanitized_path and os.path.exists(result.sanitized_path):
        sanitized_bytes = Path(result.sanitized_path).read_bytes()
        enc_sanitized = encrypt_image(sanitized_bytes, session_id)
        sanitized_stored = f"sanitized_{stored_filename}"
        (UPLOAD_DIR / sanitized_stored).write_bytes(enc_sanitized)
        os.unlink(result.sanitized_path)

    scan_data = {
        "passed": result.passed,
        "score": result.score,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks],
        "scan_ms": result.scan_ms,
    }

    return {
        "data": {
            "stored_filename": stored_filename,
            "original_filename": encrypted_image.filename,
            "scan": scan_data,
            "decrypt_ms": decrypt_ms,
        }
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
@app.get("/download/{filename:path}")
async def download_endpoint(filename: str, session_id: str = Query(None)):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    raw_bytes = file_path.read_bytes()

    if session_id:
        try:
            enc_bytes = encrypt_image(raw_bytes, session_id)
            return Response(content=enc_bytes, media_type="application/octet-stream")
        except Exception:
            pass

    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# Process (single image pipeline)
# ---------------------------------------------------------------------------
class PipelineRequest(BaseModel):
    operations: list
    session_id: str = ""


@app.post("/process")
@limiter.limit("60/minute")
async def process_endpoint(
    request: Request,
    filename: str = Query(...),
    body: PipelineRequest = None,
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if body is None:
        raise HTTPException(status_code=400, detail="Request body required")

    ops = body.operations
    session_id = body.session_id or ""
    validate_operations(ops)

    t0 = time.perf_counter()
    try:
        raw_bytes = decrypt_image(file_path.read_bytes(), session_id) if session_id else file_path.read_bytes()
    except Exception:
        raw_bytes = file_path.read_bytes()

    temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}{Path(filename).suffix}"
    temp_path.write_bytes(raw_bytes)

    try:
        result = processor_pipeline(str(temp_path), ops)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")
    finally:
        if temp_path.exists():
            os.unlink(str(temp_path))

    output_path = result["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="Processing produced no output")

    processed_bytes = Path(output_path).read_bytes()
    os.unlink(output_path)

    enc_t0 = time.perf_counter()
    if session_id:
        try:
            enc_bytes = encrypt_image(processed_bytes, session_id)
        except Exception:
            enc_bytes = processed_bytes
    else:
        enc_bytes = processed_bytes
    encrypt_ms = round((time.perf_counter() - enc_t0) * 1000, 1)

    meta = {
        "result": result,
        "processed_filename": f"processed_{filename}",
    }
    meta_b64 = base64_encode(json.dumps(meta))

    return Response(
        content=enc_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Result-Meta": meta_b64,
            "X-Encrypt-Ms": str(encrypt_ms),
        },
    )


# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------
@app.post("/batch-upload")
@limiter.limit("5/minute")
async def batch_upload(request: Request, files: list[UploadFile] = File(...), session_id: str = Form("")):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    errors = []

    for f in files[:10]:
        try:
            ext = Path(f.filename).suffix.lower() if f.filename else ".bin"
            stored_filename = f"{uuid.uuid4().hex}{ext}"
            file_path = UPLOAD_DIR / stored_filename

            raw_bytes = await f.read()
            if session_id:
                try:
                    raw_bytes = decrypt_image(raw_bytes, session_id)
                except Exception:
                    pass

            file_path.write_bytes(raw_bytes)

            scan_result = scan(str(file_path))
            results.append({
                "original_filename": f.filename,
                "stored_filename": stored_filename,
                "passed": scan_result.passed,
                "scan_score": scan_result.score,
                "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in scan_result.checks],
            })

            if scan_result.sanitized_path and os.path.exists(scan_result.sanitized_path):
                os.unlink(scan_result.sanitized_path)
        except Exception as e:
            errors.append({"filename": f.filename or "unknown", "error": str(e)})

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    return {
        "data": {
            "results": results,
            "errors": errors,
            "passed": passed_count,
            "failed": failed_count,
        }
    }


# ---------------------------------------------------------------------------
# Batch process
# ---------------------------------------------------------------------------
class BatchProcessRequest(BaseModel):
    filenames: list[str]
    operations: list
    session_id: str = ""


@app.post("/batch-process")
async def batch_process(body: BatchProcessRequest):
    filenames = body.filenames[:10]
    ops = body.operations
    session_id = body.session_id or ""
    validate_operations(ops)

    if not filenames:
        raise HTTPException(status_code=400, detail="No filenames provided")

    batch_id = uuid.uuid4().hex
    results = []
    batch_processed = []

    for fname in filenames:
        file_path = UPLOAD_DIR / fname
        if not file_path.exists():
            results.append({"original_filename": fname, "status": "error", "detail": "File not found"})
            continue

        try:
            try:
                raw_bytes = decrypt_image(file_path.read_bytes(), session_id) if session_id else file_path.read_bytes()
            except Exception:
                raw_bytes = file_path.read_bytes()

            temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}{Path(fname).suffix}"
            temp_path.write_bytes(raw_bytes)

            pipe_result = processor_pipeline(str(temp_path), ops)
            output_path = pipe_result["output_path"]

            if os.path.exists(output_path):
                out_bytes = Path(output_path).read_bytes()
                out_stored = f"processed_{fname}"
                (UPLOAD_DIR / out_stored).write_bytes(out_bytes)
                os.unlink(output_path)
                batch_processed.append(out_stored)

            results.append({"original_filename": fname, "status": "success"})

            if temp_path.exists():
                os.unlink(str(temp_path))
        except Exception as e:
            results.append({"original_filename": fname, "status": "error", "detail": str(e)})

    _batch_files[batch_id] = batch_processed

    return {
        "data": {
            "batch_id": batch_id,
            "results": results,
        }
    }


# ---------------------------------------------------------------------------
# Batch download (ZIP)
# ---------------------------------------------------------------------------
@app.get("/batch-download/{batch_id}")
async def batch_download(batch_id: str):
    files = _batch_files.get(batch_id)
    if not files:
        raise HTTPException(status_code=404, detail="Batch not found or expired")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            f = UPLOAD_DIR / name
            if f.exists():
                zf.write(str(f), arcname=name)
    buf.seek(0)
    return Response(content=buf.read(), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.zip"})


# ---------------------------------------------------------------------------
# Static frontend files (must be after all API routes)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def base64_encode(s: str) -> str:
    return __import__("base64").b64encode(s.encode()).decode()
