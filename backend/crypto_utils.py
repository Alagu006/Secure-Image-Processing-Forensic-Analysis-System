import json
import os
import base64
import time
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SESSION_TTL = 3600
_USE_REDIS = os.environ.get("REDIS_URL") is not None

if _USE_REDIS:
    import redis
    _redis_client = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
else:
    _sessions = {}


def _set_session(session_id: str, key_bytes: bytes):
    payload = {
        "key_b64": base64.b64encode(key_bytes).decode(),
        "created": time.time(),
    }
    if _USE_REDIS:
        _redis_client.setex(f"session:{session_id}", SESSION_TTL, json.dumps(payload))
    else:
        _sessions[session_id] = payload
        _cleanup_expired()


def _get_session(session_id: str) -> dict:
    if _USE_REDIS:
        raw = _redis_client.get(f"session:{session_id}")
        if not raw:
            raise ValueError("Invalid or expired session")
        return json.loads(raw)
    else:
        session = _sessions.get(session_id)
        if not session:
            raise ValueError("Invalid or expired session")
        if time.time() - session["created"] > SESSION_TTL:
            del _sessions[session_id]
            raise ValueError("Session expired")
        return session


def _delete_session(session_id: str):
    if _USE_REDIS:
        _redis_client.delete(f"session:{session_id}")
    else:
        _sessions.pop(session_id, None)


def create_session():
    session_id = os.urandom(16).hex()
    key = os.urandom(32)
    _set_session(session_id, key)
    return {"session_id": session_id, "key_b64": base64.b64encode(key).decode()}


def get_session_key(session_id: str) -> bytes:
    session = _get_session(session_id)
    return base64.b64decode(session["key_b64"])


def decrypt_image(encrypted_bytes: bytes, session_id: str) -> bytes:
    key = get_session_key(session_id)
    iv = encrypted_bytes[:12]
    ciphertext_and_tag = encrypted_bytes[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ciphertext_and_tag, None)


def encrypt_image(raw_bytes: bytes, session_id: str) -> bytes:
    key = get_session_key(session_id)
    iv = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(iv, raw_bytes, None)
    return iv + ciphertext_and_tag


def _cleanup_expired():
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["created"] > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
