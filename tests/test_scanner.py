import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from backend.scanner import scan


def _cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass


def _sanitized_path(original: str) -> str:
    p = Path(original)
    return str(p.with_name(p.stem + "_sanitized" + p.suffix))


@pytest.fixture(autouse=True)
def cleanup_sanitized(request):
    yield
    marker = request.node.get_closest_marker("cleanup_path")
    if marker:
        for path in marker.args:
            _cleanup(path, _sanitized_path(path))


def test_valid_jpeg_passes():
    img = Image.new("RGB", (100, 100), color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG")
        tmp = f.name
    try:
        result = scan(tmp)
        assert result.passed, f"Valid JPEG should pass, got score={result.score}"
    finally:
        _cleanup(tmp, _sanitized_path(tmp))


def test_invalid_magic_bytes_fails():
    data = b"\x00\x01\x02\x03" + b"\x00" * 96
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="does not appear to be a raw image"):
            scan(tmp)
    finally:
        _cleanup(tmp)


def test_large_file_rejected():
    img = Image.new("RGB", (10, 10), color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG")
        tmp = f.name
    try:
        with patch("backend.scanner.MAX_FILE_SIZE", 10):
            result = scan(tmp)
        size_check = [c for c in result.checks if c.name == "file_size"][0]
        assert not size_check.passed, "File exceeding size limit should be rejected"
    finally:
        _cleanup(tmp, _sanitized_path(tmp))


def test_entropy_score():
    # PNG magic + zeroed data (magic bytes needed for guard, rest is zeros)
    data = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a" + b"\x00" * 4088
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        result = scan(tmp)
        entropy_check = [c for c in result.checks if c.name == "entropy"][0]
        match = re.search(r"([\d.]+)", entropy_check.detail)
        assert match, f"Could not extract entropy from: {entropy_check.detail}"
        value = float(match.group(1))
        assert 0.0 <= value <= 8.0, f"Entropy {value} outside expected range [0, 8]"
    finally:
        _cleanup(tmp)


def test_exif_stripped_after_reencoding():
    img = Image.new("RGB", (50, 50), color="blue")
    exif = img.getexif()
    exif[271] = "Test Camera"
    exif[272] = "Test Model"
    exif_bytes = exif.tobytes()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, "JPEG", exif=exif_bytes)
        tmp = f.name

    try:
        result = scan(tmp)
        assert result.sanitized_path, "Sanitized file was not produced"
        sanitized_img = Image.open(result.sanitized_path)
        sanitized_exif = sanitized_img._getexif()
        assert sanitized_exif is None or len(sanitized_exif) == 0, (
            "EXIF data was not stripped during re-encoding"
        )
    finally:
        _cleanup(tmp, result.sanitized_path if result and result.sanitized_path else None)


def test_operation_validation_rejects_unknown_op():
    from backend.main import validate_operations, VALID_OPERATIONS
    with pytest.raises(Exception) as exc:
        validate_operations([{"operation": "rm -rf", "params": {}}])
    err = exc.value
    assert err.status_code == 422
    assert "rm -rf" in err.detail["error"]
    assert err.detail["valid_operations"] == VALID_OPERATIONS


def test_operation_validation_empty():
    from backend.main import validate_operations, VALID_OPERATIONS
    with pytest.raises(Exception) as exc:
        validate_operations([])
    assert exc.value.status_code == 400


def test_operation_validation_valid():
    from backend.main import validate_operations
    validate_operations([{"operation": "rotate", "params": {"angle": 45}}])
    validate_operations([{"operation": "grayscale"}])
    validate_operations(["grayscale", {"operation": "rotate", "params": {"angle": 90}}])


def test_operation_validation_bad_param_type():
    from backend.main import validate_operations
    with pytest.raises(Exception) as exc:
        validate_operations([{"operation": "rotate", "params": {"angle": "not_a_number"}}])
    assert exc.value.status_code == 422


def test_operation_validation_param_out_of_range():
    from backend.main import validate_operations
    with pytest.raises(Exception) as exc:
        validate_operations([{"operation": "rotate", "params": {"angle": 999}}])
    assert exc.value.status_code == 422


def test_redis_fallback():
    import os
    orig = os.environ.pop("REDIS_URL", None)
    try:
        from backend.crypto_utils import create_session, get_session_key
        session = create_session()
        assert "session_id" in session
        key = get_session_key(session["session_id"])
        assert len(key) == 32
    finally:
        if orig is not None:
            os.environ["REDIS_URL"] = orig


def test_redis_fallback_encrypt_decrypt():
    import os
    orig = os.environ.pop("REDIS_URL", None)
    try:
        from backend.crypto_utils import create_session, encrypt_image, decrypt_image
        session = create_session()
        sid = session["session_id"]
        data = b"test data for redis fallback"
        encrypted = encrypt_image(data, sid)
        decrypted = decrypt_image(encrypted, sid)
        assert decrypted == data
    finally:
        if orig is not None:
            os.environ["REDIS_URL"] = orig
