import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

try:
    from utils import MAX_FILE_SIZE
except ImportError:
    from .utils import MAX_FILE_SIZE

IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89\x50\x4e\x47": "PNG",
    b"\x47\x49\x46\x38": "GIF",
    b"\x52\x49\x46\x46": "WEBP (RIFF)",
}

POLYGLOT_SIGNATURES = [
    (b"PK\x03\x04", "ZIP"),
    (b"%PDF", "PDF"),
    (b"\x7fELF", "ELF"),
    (b"MZ", "MZ (PE)"),
    (b"\x1f\x8b", "GZIP"),
    (b"BZ", "BZ2"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "RAR"),
    (b"<?xml", "XML"),
]


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "", ms: float = 0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.ms = round(ms, 1)

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "ms": self.ms}


class ScanResult:
    def __init__(
        self,
        passed: bool,
        score: int,
        checks: list[CheckResult],
        sanitized_path: str = "",
        file_info: dict = None,
        scan_ms: float = 0,
    ):
        self.passed = passed
        self.score = score
        self.checks = checks
        self.sanitized_path = sanitized_path
        self.file_info = file_info or {}
        self.scan_ms = round(scan_ms, 1)

    def to_dict(self):
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [c.to_dict() for c in self.checks],
            "sanitized_path": self.sanitized_path,
            "file_info": self.file_info,
            "scan_ms": self.scan_ms,
        }


def _timed(check_fn):
    """Decorator that records execution time in ms on the returned CheckResult."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = check_fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        result.ms = round(elapsed, 1)
        return result
    return wrapper


def scan(file_path: str) -> ScanResult:
    t0 = time.perf_counter()

    # Guard: verify file starts with known image magic bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        is_image = any(header.startswith(magic) for magic in IMAGE_MAGIC_BYTES)
        if not is_image:
            raise ValueError(
                "File does not appear to be a raw image — "
                "possible encryption/encoding issue upstream"
            )
    except OSError as e:
        raise ValueError(f"Cannot read file for magic byte verification: {e}")

    file_info = _gather_file_info(file_path)

    # Phase 1 — fast fail (sequential, microsecond ops)
    checks = []
    checks.append(_check_magic_bytes(file_path))
    checks.append(_check_file_size(file_path))
    checks.append(_check_pillow_header(file_path))

    # Phase 2 — parallel (entropy + polyglot are I/O + CPU heavy)
    parallel_checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_poly = pool.submit(_check_polyglot, file_path)
        fut_ent = pool.submit(_check_entropy, file_path)
        for fut in as_completed([fut_poly, fut_ent]):
            parallel_checks.append(fut.result())
    checks.extend(sorted(parallel_checks, key=lambda c: c.name))

    # Phase 3 — remaining sequential
    checks.append(_check_exif(file_path))
    checks.append(_check_dimensions(file_path))
    checks.append(_pixel_re_encode(file_path))

    passed_count = sum(1 for c in checks if c.passed)
    total = len(checks)
    score = int((passed_count / total) * 100) if total else 0
    overall_passed = score >= 75

    sanitized_path = ""
    for c in checks:
        if c.name == "pixel_re_encoding" and c.passed:
            sanitized_path = c.detail
            break

    scan_ms = (time.perf_counter() - t0) * 1000
    return ScanResult(
        passed=overall_passed,
        score=score,
        checks=checks,
        sanitized_path=sanitized_path,
        file_info=file_info,
        scan_ms=scan_ms,
    )


def _gather_file_info(file_path: str) -> dict:
    try:
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "extension": Path(file_path).suffix.lower(),
            "filename": Path(file_path).name,
        }
    except OSError:
        return {}


# ---------------------------------------------------------------------------
# Helper: detect image format from file header
# ---------------------------------------------------------------------------
def _detect_format(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            h = f.read(16)
    except OSError:
        return "UNKNOWN"
    if h[:2] == b"\xff\xd8":
        return "JPEG"
    if h[:4] == b"\x89\x50\x4e\x47":
        return "PNG"
    if h[:3] == b"\x47\x49\x46":
        return "GIF"
    if h[:4] == b"\x52\x49\x46\x46":
        return "WEBP"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Magic bytes
# ---------------------------------------------------------------------------
@_timed
def _check_magic_bytes(file_path: str) -> CheckResult:
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        if len(header) < 4:
            return CheckResult("magic_bytes", False, "File too small to read header")
        for magic, name in IMAGE_MAGIC_BYTES.items():
            if header.startswith(magic):
                detail = f"Matched {name} signature"
                if name == "WEBP (RIFF)":
                    if len(header) >= 12 and header[8:12] == b"WEBP":
                        detail = "Matched WEBP signature"
                    else:
                        return CheckResult("magic_bytes", False, "RIFF header but not a valid WEBP")
                return CheckResult("magic_bytes", True, detail)
        hex_str = header[:8].hex(" ").upper()
        return CheckResult("magic_bytes", False, f"Unknown image magic bytes: {hex_str}")
    except OSError as e:
        return CheckResult("magic_bytes", False, f"IO error reading file: {e}")


# ---------------------------------------------------------------------------
# File size
# ---------------------------------------------------------------------------
@_timed
def _check_file_size(file_path: str) -> CheckResult:
    try:
        size = os.path.getsize(file_path)
        if size > MAX_FILE_SIZE:
            return CheckResult("file_size", False, f"File size {size} bytes exceeds {MAX_FILE_SIZE} byte limit")
        return CheckResult("file_size", True, f"File size {size} bytes within limit")
    except OSError as e:
        return CheckResult("file_size", False, f"Could not determine file size: {e}")


# ---------------------------------------------------------------------------
# Pillow header verification
# ---------------------------------------------------------------------------
@_timed
def _check_pillow_header(file_path: str) -> CheckResult:
    try:
        img = Image.open(file_path)
        img.verify()
        return CheckResult("pillow_header", True, "Pillow opened and verified image")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return CheckResult("pillow_header", False, f"Pillow rejected image: {e}")


# ---------------------------------------------------------------------------
# Entropy — format-aware thresholds, warnings not hard fails
# ---------------------------------------------------------------------------
_entropy_cache: dict[int, float] = {}


@_timed
def _check_entropy(file_path: str) -> CheckResult:
    try:
        fmt = _detect_format(file_path)
        threshold = {"JPEG": 7.98, "PNG": 7.90}.get(fmt, 7.95)

        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return CheckResult("entropy", False, "Empty file, cannot compute entropy")

        data_hash = hash(data[:4096])
        if data_hash in _entropy_cache:
            entropy = _entropy_cache[data_hash]
        else:
            freq = [0] * 256
            for byte in data:
                freq[byte] += 1
            entropy = 0.0
            length = len(data)
            for count in freq:
                if count:
                    p = count / length
                    entropy -= p * math.log2(p)
            entropy = round(entropy, 4)
            if len(_entropy_cache) < 128:
                _entropy_cache[data_hash] = entropy

        if entropy == 8.0:
            return CheckResult("entropy", False, f"Entropy exactly 8.0 — file appears to contain random/encrypted data")
        if entropy > threshold:
            return CheckResult("entropy", True, f"High entropy warning ({entropy}) for {fmt} — review recommended")
        return CheckResult("entropy", True, f"Entropy {entropy} within normal range for {fmt}")
    except OSError as e:
        return CheckResult("entropy", False, f"Could not compute entropy: {e}")


# ---------------------------------------------------------------------------
# Polyglot — last 20%, JPEG EOI strip, validated MZ/ZIP
# ---------------------------------------------------------------------------
def _has_valid_mz_pe(data: bytes, pos: int) -> bool:
    """Check for full DOS header: MZ magic + valid e_lfanew at offset 0x3C."""
    if pos + 0x40 + 4 > len(data):
        return False
    pe_off = int.from_bytes(data[pos + 0x3C:pos + 0x40], "little")
    return 64 <= pe_off <= 512


def _has_valid_zip_local(data: bytes, pos: int) -> bool:
    """Check for complete ZIP local file header with plausible version."""
    if pos + 30 > len(data):
        return False
    version = int.from_bytes(data[pos + 4:pos + 6], "little")
    return 10 <= version <= 63


@_timed
def _check_polyglot(file_path: str) -> CheckResult:
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if len(data) < 512:
            return CheckResult("polyglot", True, "File too small for polyglot scan")

        # Determine search boundaries
        is_jpeg = data[:2] == b"\xff\xd8"
        search_end = len(data)

        # For JPEG, strip everything after EOI marker (FF D9)
        if is_jpeg:
            eoi = data.find(b"\xff\xd9")
            if eoi != -1:
                search_end = eoi + 2

        # Search region: skip first 512 bytes, also enforce JPEG EOI boundary
        payload = data[512:search_end]
        if not payload:
            return CheckResult("polyglot", True, "No payload region to scan")

        # Only scan last 20% of the payload (real polyglots append at the end)
        scan_start = int(len(payload) * 0.8)
        body = payload[scan_start:]

        found = []
        for sig, name in POLYGLOT_SIGNATURES:
            if name == "MZ (PE)":
                idx = 0
                while True:
                    idx = body.find(b"MZ", idx)
                    if idx == -1:
                        break
                    if _has_valid_mz_pe(body, idx):
                        found.append(name)
                        break
                    idx += 1
            elif name == "ZIP":
                idx = 0
                while True:
                    idx = body.find(b"PK\x03\x04", idx)
                    if idx == -1:
                        break
                    if _has_valid_zip_local(body, idx):
                        found.append(name)
                        break
                    idx += 1
            else:
                if sig in body:
                    found.append(name)

        if found:
            return CheckResult("polyglot", False, f"Embedded format signatures detected: {', '.join(found)}")
        return CheckResult("polyglot", True, "No polyglot signatures found")
    except OSError as e:
        return CheckResult("polyglot", False, f"IO error scanning for polyglot: {e}")


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------
@_timed
def _check_exif(file_path: str) -> CheckResult:
    try:
        with Image.open(file_path) as img:
            exif_data = img.getexif()
    except Exception as e:
        return CheckResult("exif", True, f"No EXIF data accessible: {e}")

    if exif_data is None or len(exif_data) == 0:
        return CheckResult("exif", True, "No EXIF metadata found")

    exif_keys = list(exif_data.keys())
    has_gps = 0x8825 in exif_data

    if has_gps:
        return CheckResult("exif", False, f"GPS location data present in EXIF. Keys: {exif_keys}")
    return CheckResult("exif", True, f"EXIF metadata found ({len(exif_keys)} tags). No GPS data. Keys: {exif_keys}")


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
@_timed
def _check_dimensions(file_path: str) -> CheckResult:
    try:
        with Image.open(file_path) as img:
            width, height = img.size
        if width > 10000 or height > 10000:
            return CheckResult("dimensions", False, f"Image too large: {width}x{height} (max 10000x10000)")
        if width < 10 or height < 10:
            return CheckResult("dimensions", False, f"Image too small: {width}x{height} (min 10x10)")
        return CheckResult("dimensions", True, f"Dimensions {width}x{height} within limits")
    except Exception as e:
        return CheckResult("dimensions", False, f"Could not read dimensions: {e}")


# ---------------------------------------------------------------------------
# Pixel re-encoding / sanitization
# ---------------------------------------------------------------------------
@_timed
def _pixel_re_encode(file_path: str) -> CheckResult:
    try:
        with Image.open(file_path) as img:
            img.load()
        orig_path = Path(file_path)

        # Use correct extension based on actual PIL format, not file suffix
        pil_fmt = (img.format or "PNG").upper()
        ext_map = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff", "WEBP": ".webp"}
        ext = ext_map.get(pil_fmt, f".{pil_fmt.lower()}")

        sanitized_name = f"{orig_path.stem}_sanitized{ext}"
        sanitized_path = orig_path.with_name(sanitized_name)
        save_fmt = "JPEG" if pil_fmt == "JPG" else pil_fmt
        if img.mode in ("P", "L", "LA", "PA"):
            img = img.convert("RGBA")
        img.save(str(sanitized_path), format=save_fmt)
        return CheckResult("pixel_re_encoding", True, str(sanitized_path))
    except Exception as e:
        err_str = str(e)
        if "ENC" in err_str or "cannot identify" in err_str:
            return CheckResult(
                "pixel_re_encoding", False,
                "Received encrypted or corrupted data — ensure image is decrypted before scanning",
            )
        return CheckResult("pixel_re_encoding", False, f"Re-encoding failed: {e}")


scan_file = scan
