"""
auth.py — Authentication helpers for AudioForensics AI
Handles password hashing, validation, and Streamlit session management.
"""

import re
import hashlib
import hmac
import os
import streamlit as st

from database import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    username_exists,
    email_exists,
)

# ── A random app-level secret stored in env or a local file ──────────────────
# In production set the env var:  export AF_SECRET="your-random-secret"
_SECRET = os.environ.get("AF_SECRET", "audioforensics-dev-secret-change-in-prod")


# ===============================
# PASSWORD HASHING  (PBKDF2-HMAC-SHA256)
# ===============================
def _hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'salt_hex$hash_hex' string."""
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + "$" + key.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    try:
        salt_hex, _ = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = _hash_password(password, salt)
        return hmac.compare_digest(expected, stored_hash)
    except Exception:
        return False


# ===============================
# VALIDATION
# ===============================
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def validate_signup(username: str, email: str, password: str, confirm: str) -> list[str]:
    """Return list of error strings (empty list = OK)."""
    errors = []
    username = username.strip()
    email = email.strip()

    if len(username) < 3:
        errors.append("Username must be at least 3 characters.")
    elif len(username) > 30:
        errors.append("Username must be 30 characters or fewer.")
    elif not re.match(r"^[a-zA-Z0-9_]+$", username):
        errors.append("Username may only contain letters, numbers, and underscores.")
    elif username_exists(username):
        errors.append("Username is already taken.")

    if not _EMAIL_RE.match(email):
        errors.append("Enter a valid email address.")
    elif email_exists(email):
        errors.append("An account with this email already exists.")

    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    elif password != confirm:
        errors.append("Passwords do not match.")

    return errors


# ===============================
# SIGNUP / LOGIN
# ===============================
def signup_user(username: str, email: str, password: str) -> int:
    """Hash password and persist user. Returns new user id."""
    hashed = _hash_password(password)
    return create_user(username.strip(), email.strip().lower(), hashed)


def login_user(username_or_email: str, password: str):
    """
    Try to authenticate by username OR email.
    Returns user row dict on success, None on failure.
    """
    val = username_or_email.strip()
    if "@" in val:
        user = get_user_by_email(val)
    else:
        user = get_user_by_username(val)

    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return dict(user)


# ===============================
# SESSION HELPERS
# ===============================
SESSION_KEY = "af_user"   # key stored in st.session_state


def set_session(user: dict):
    """Store user info in Streamlit session state."""
    st.session_state[SESSION_KEY] = {
        "id":       user["id"],
        "username": user["username"],
        "email":    user["email"],
    }


def get_session() -> dict | None:
    """Return current user dict or None if not logged in."""
    return st.session_state.get(SESSION_KEY)


def is_logged_in() -> bool:
    return get_session() is not None


def logout():
    st.session_state.pop(SESSION_KEY, None)
    # Also clear any analysis context
    for k in ["current_analysis_id", "active_page"]:
        st.session_state.pop(k, None)