import json
import uuid
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
BATCH_DIR = BASE_DIR / "batch_manifests"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_FILES = 10


def generate_uuid_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_upload_path(filename: str) -> Path:
    return UPLOAD_DIR / filename


def get_processed_path(filename: str) -> Path:
    return PROCESSED_DIR / filename


def get_batch_path(batch_id: str) -> Path:
    return BATCH_DIR / f"{batch_id}.json"


def save_batch_manifest(batch_id: str, data: dict):
    path = get_batch_path(batch_id)
    with open(path, "w") as f:
        json.dump(data, f)


def load_batch_manifest(batch_id: str) -> dict:
    path = get_batch_path(batch_id)
    with open(path) as f:
        return json.load(f)


def clean_filename(filename: str) -> str:
    return Path(filename).name
