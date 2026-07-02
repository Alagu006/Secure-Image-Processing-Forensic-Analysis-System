import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.crypto_utils import create_session, get_session_key, decrypt_image, encrypt_image


def test_create_session_returns_key_and_id():
    session = create_session()
    assert "session_id" in session
    assert "key_b64" in session
    assert len(session["session_id"]) == 32  # 16 bytes hex
    assert len(session["key_b64"]) > 0


def test_get_session_key_valid():
    session = create_session()
    key = get_session_key(session["session_id"])
    assert len(key) == 32


def test_get_session_key_invalid():
    with pytest.raises(ValueError, match="Invalid or expired"):
        get_session_key("nonexistent")


def test_encrypt_decrypt_roundtrip():
    session = create_session()
    sid = session["session_id"]
    original = b"hello world this is test data " * 100
    encrypted = encrypt_image(original, sid)
    decrypted = decrypt_image(encrypted, sid)
    assert decrypted == original


def test_encrypted_data_is_different():
    session = create_session()
    sid = session["session_id"]
    original = b"test data"
    encrypted = encrypt_image(original, sid)
    assert encrypted != original
    assert len(encrypted) > len(original)


def test_decrypt_wrong_session_fails():
    s1 = create_session()
    s2 = create_session()
    encrypted = encrypt_image(b"secret", s1["session_id"])
    with pytest.raises(Exception):
        decrypt_image(encrypted, s2["session_id"])


def test_decrypt_tampered_data_fails():
    session = create_session()
    sid = session["session_id"]
    encrypted = encrypt_image(b"secret", sid)
    tampered = bytearray(encrypted)
    tampered[5] ^= 0xFF
    with pytest.raises(Exception):
        decrypt_image(bytes(tampered), sid)


def test_multiple_sessions_independent():
    s1 = create_session()
    s2 = create_session()
    data1 = encrypt_image(b"data1", s1["session_id"])
    data2 = encrypt_image(b"data2", s2["session_id"])
    assert decrypt_image(data1, s1["session_id"]) == b"data1"
    assert decrypt_image(data2, s2["session_id"]) == b"data2"
