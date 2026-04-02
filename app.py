from __future__ import annotations


import base64
import hashlib
import hmac
import inspect
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------
# App configuration
# -------------------------------

APP_TITLE = "NeuraNexus"
APP_TAGLINE = "Brainwave-Assisted Mental Wellness and Cognitive Monitoring"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "neuraneux.db"

PASSWORD_ITERATIONS = 210_000

SIM_SAMPLE_RATE_HZ = 256
SIM_WINDOW_SECONDS = 2.0

STRESS_THRESHOLD_HIGH = 1.5
STRESS_THRESHOLD_SEVERE = 2.0


# -------------------------------
# Utilities: Streamlit compat
# -------------------------------

def _rerun() -> None:
    """Rerun the Streamlit script (compat across Streamlit versions)."""

    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


def _toast(message: str, icon: Optional[str] = None) -> None:
    """Show a toast notification when supported."""

    if hasattr(st, "toast"):
        # Streamlit's toast accepts icon as emoji (or None).
        st.toast(message, icon=icon)


def _wide_kwargs(widget_fn: Any) -> Dict[str, Any]:
    """Return keyword args for a full-width widget across Streamlit versions.

    Streamlit is deprecating `use_container_width` in favor of `width='stretch'`.
    This helper keeps the app clean on newer versions while remaining compatible
    with older ones.
    """

    try:
        params = inspect.signature(widget_fn).parameters
    except (TypeError, ValueError):
        return {}

    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# -------------------------------
# Database layer (SQLite)
# -------------------------------


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection with safe defaults."""

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""

    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_iterations INTEGER NOT NULL,
                age INTEGER,
                emergency_contact TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eeg_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                theta REAL NOT NULL,
                stress_index REAL NOT NULL,
                status TEXT NOT NULL,
                severity TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eeg_logs_user_ts
            ON eeg_logs(user_id, timestamp)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                date TEXT NOT NULL,
                time_slot TEXT NOT NULL,
                consultation_type TEXT NOT NULL,
                notes TEXT,
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_appointments_user_date
            ON appointments(user_email, date)
            """
        )


# -------------------------------
# Authentication: secure hashing
# -------------------------------


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("utf-8"))


def hash_password(password: str, *, iterations: int = PASSWORD_ITERATIONS) -> Tuple[str, str, int]:
    """Hash a password using PBKDF2-HMAC-SHA256 with a random per-user salt."""

    if not password:
        raise ValueError("Password cannot be empty")

    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return _b64encode(derived), _b64encode(salt), iterations


def verify_password(password: str, *, salt_b64: str, hash_b64: str, iterations: int) -> bool:
    """Constant-time password verification."""

    try:
        salt = _b64decode(salt_b64)
        expected = _b64decode(hash_b64)
    except Exception:
        return False

    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return hmac.compare_digest(derived, expected)


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def create_user(
    *,
    name: str,
    email: str,
    password: str,
    age: Optional[int],
    emergency_contact: str,
) -> Tuple[bool, str]:
    """Create a new user. Returns (success, message)."""

    name = (name or "").strip()
    email = (email or "").strip().lower()
    emergency_contact = (emergency_contact or "").strip()

    if not name:
        return False, "Name is required."
    if not email or not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters."
    if age is not None and (age < 10 or age > 120):
        return False, "Please enter a realistic age."
    if not emergency_contact:
        return False, "Emergency contact is required."

    pwd_hash, pwd_salt, iters = hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO users (name, email, password_hash, password_salt, password_iterations, age, emergency_contact, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, email, pwd_hash, pwd_salt, iters, age, emergency_contact, created_at),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That email is already registered."
    except Exception:
        return False, "Could not create account. Please try again."


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    email = (email or "").strip().lower()
    if not email:
        return None

    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return row


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
    return row


def authenticate(email: str, password: str) -> Tuple[bool, str, Optional[sqlite3.Row]]:
    """Authenticate user by email/password. Returns (success, message, user)."""

    user = get_user_by_email(email)
    if not user:
        return False, "Invalid email or password.", None

    ok = verify_password(
        password,
        salt_b64=user["password_salt"],
        hash_b64=user["password_hash"],
        iterations=int(user["password_iterations"]),
    )
    if not ok:
        return False, "Invalid email or password.", None

    return True, "Logged in.", user


# -------------------------------
# EEG simulation + signal processing
# -------------------------------


@dataclass
class EEGSimState:
    rng: np.random.Generator
    stress_drive: float

    base_alpha_uv: float
    base_beta_uv: float
    base_theta_uv: float

    phase_alpha: float
    phase_beta: float
    phase_theta: float


def get_sim_state() -> EEGSimState:
    """Get or initialize a per-session EEG simulation state."""

    state = st.session_state.get("sim_state")
    if state is not None:
        return state

    seed = secrets.randbelow(2**32 - 1)
    rng = np.random.default_rng(seed)

    state = EEGSimState(
        rng=rng,
        stress_drive=float(rng.uniform(0.15, 0.35)),
        base_alpha_uv=float(rng.uniform(22, 30)),
        base_beta_uv=float(rng.uniform(12, 18)),
        base_theta_uv=float(rng.uniform(18, 26)),
        phase_alpha=float(rng.uniform(0, 2 * np.pi)),
        phase_beta=float(rng.uniform(0, 2 * np.pi)),
        phase_theta=float(rng.uniform(0, 2 * np.pi)),
    )

    st.session_state["sim_state"] = state
    return state


def _bandpower_mean(psd: np.ndarray, freqs: np.ndarray, f_low: float, f_high: float) -> float:
    idx = (freqs >= f_low) & (freqs < f_high)
    if not np.any(idx):
        return 0.0
    return float(np.mean(psd[idx]))


def compute_band_powers(signal: np.ndarray, fs_hz: int) -> Tuple[float, float, float]:
    """Compute average band power for Theta/Alpha/Beta bands using FFT."""

    signal = signal - float(np.mean(signal))
    n = int(signal.shape[0])

    window = np.hanning(n)
    xw = signal * window

    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    fft = np.fft.rfft(xw)

    # Power spectrum (relative units); mean band power suffices for ratio metrics.
    psd = (np.abs(fft) ** 2) / float(n)

    theta = _bandpower_mean(psd, freqs, 4.0, 8.0)
    alpha = _bandpower_mean(psd, freqs, 8.0, 12.0)
    beta = _bandpower_mean(psd, freqs, 13.0, 30.0)
    return alpha, beta, theta


def classify_stress(stress_index: float) -> Tuple[str, str, str]:
    """Classify stress by rule-based thresholds.

    Returns: (status, severity, badge_emoji)
    """

    if stress_index > STRESS_THRESHOLD_SEVERE:
        return "HIGH", "High", "🔴"
    if stress_index > STRESS_THRESHOLD_HIGH:
        return "HIGH", "Moderate", "🟠"
    return "NORMAL", "Normal", "🟢"


def simulate_eeg_window(
    fs_hz: int = SIM_SAMPLE_RATE_HZ,
    seconds: float = SIM_WINDOW_SECONDS,
    *,
    stress_level: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a synthetic EEG-like time series for one window and return metadata."""

    state = get_sim_state()
    rng = state.rng

    n = int(fs_hz * seconds)
    t = np.arange(n, dtype=float) / float(fs_hz)

    # Stress drive:
    # - If stress_level is provided, use it (manual simulation).
    # - Otherwise evolve naturally with a random-walk drift.
    if stress_level is None:
        state.stress_drive = float(np.clip(state.stress_drive + rng.normal(0.0, 0.08), 0.0, 1.0))
        if rng.random() < 0.04:
            state.stress_drive = float(np.clip(state.stress_drive + rng.uniform(0.35, 0.75), 0.0, 1.0))
        if rng.random() < 0.03:
            state.stress_drive = float(np.clip(state.stress_drive - rng.uniform(0.20, 0.55), 0.0, 1.0))
    else:
        state.stress_drive = float(np.clip(float(stress_level), 0.0, 1.0))

    # Small baseline drifts (microvolts).
    state.base_alpha_uv = float(np.clip(state.base_alpha_uv + rng.normal(0.0, 0.6), 18.0, 35.0))
    state.base_beta_uv = float(np.clip(state.base_beta_uv + rng.normal(0.0, 0.6), 10.0, 28.0))
    state.base_theta_uv = float(np.clip(state.base_theta_uv + rng.normal(0.0, 0.6), 12.0, 32.0))

    # Frequency jitter within each band.
    theta_f = float(np.clip(rng.normal(6.0, 0.4), 4.0, 8.0))
    alpha_f = float(np.clip(rng.normal(10.0, 0.5), 8.0, 12.0))
    beta_f = float(np.clip(rng.normal(20.0, 2.0), 13.0, 30.0))

    # Modulate amplitudes by stress: beta rises, alpha dips.
    alpha_amp = float(max(5.0, state.base_alpha_uv * (1.0 - 0.35 * state.stress_drive) + rng.normal(0.0, 0.8)))
    beta_amp = float(max(5.0, state.base_beta_uv * (1.0 + 1.20 * state.stress_drive) + rng.normal(0.0, 0.8)))
    theta_amp = float(max(5.0, state.base_theta_uv * (1.0 + 0.20 * (1.0 - state.stress_drive)) + rng.normal(0.0, 0.8)))

    # Use current phases for this window; then advance for continuity.
    phase_theta = float(state.phase_theta)
    phase_alpha = float(state.phase_alpha)
    phase_beta = float(state.phase_beta)

    noise = rng.normal(0.0, 6.0, size=n)
    signal = (
        theta_amp * np.sin(2.0 * np.pi * theta_f * t + phase_theta)
        + alpha_amp * np.sin(2.0 * np.pi * alpha_f * t + phase_alpha)
        + beta_amp * np.sin(2.0 * np.pi * beta_f * t + phase_beta)
        + noise
    )

    state.phase_theta = float((phase_theta + 2.0 * np.pi * theta_f * seconds) % (2.0 * np.pi))
    state.phase_alpha = float((phase_alpha + 2.0 * np.pi * alpha_f * seconds) % (2.0 * np.pi))
    state.phase_beta = float((phase_beta + 2.0 * np.pi * beta_f * seconds) % (2.0 * np.pi))

    meta = {
        "theta_f": theta_f,
        "alpha_f": alpha_f,
        "beta_f": beta_f,
        "stress_drive": state.stress_drive,
        "alpha_amp": alpha_amp,
        "beta_amp": beta_amp,
        "theta_amp": theta_amp,
    }
    return signal.astype(float), meta


def generate_reading(*, stress_level: Optional[float] = None) -> Dict[str, Any]:
    """Generate one EEG reading (band powers + stress classification)."""

    signal, meta = simulate_eeg_window(stress_level=stress_level)
    alpha, beta, theta = compute_band_powers(signal, SIM_SAMPLE_RATE_HZ)

    stress_index = float(beta / max(alpha, 1e-9))
    status, severity, badge = classify_stress(stress_index)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "timestamp": ts,
        "alpha": float(alpha),
        "beta": float(beta),
        "theta": float(theta),
        "stress_index": float(stress_index),
        "status": status,
        "severity": severity,
        "badge": badge,
        "meta": meta,
    }


def log_reading(user_id: int, reading: Dict[str, Any]) -> None:
    """Persist a reading into SQLite."""

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO eeg_logs (user_id, timestamp, alpha, beta, theta, stress_index, status, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                reading["timestamp"],
                float(reading["alpha"]),
                float(reading["beta"]),
                float(reading["theta"]),
                float(reading["stress_index"]),
                str(reading["status"]),
                str(reading["severity"]),
            ),
        )


def fetch_history(user_id: int, *, limit: int = 2000) -> pd.DataFrame:
    """Load history for a user from SQLite."""

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, alpha, beta, theta, stress_index, status, severity
            FROM eeg_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp", "alpha", "beta", "theta", "stress_index", "status", "severity"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp")
    return df


# -------------------------------
# Appointments (SQLite)
# -------------------------------


TIME_SLOTS: Tuple[str, ...] = ("Morning", "Afternoon", "Evening")
CONSULTATION_TYPES: Tuple[str, ...] = ("Stress Counseling", "Mental Wellness", "Emergency Support")
APPOINTMENT_STATUSES: Tuple[str, ...] = ("Scheduled", "Completed", "Cancelled")


def create_appointment(
    *,
    user_email: str,
    appt_date: str,
    time_slot: str,
    consultation_type: str,
    notes: str,
    status: str = "Scheduled",
) -> Tuple[bool, str, Optional[int]]:
    """Create an appointment. Returns (success, message, appointment_id)."""

    user_email = (user_email or "").strip().lower()
    appt_date = (appt_date or "").strip()
    time_slot = (time_slot or "").strip()
    consultation_type = (consultation_type or "").strip()
    notes = (notes or "").strip()
    status = (status or "").strip()

    if not user_email or not EMAIL_RE.match(user_email):
        return False, "Invalid user email.", None
    if not appt_date:
        return False, "Please select a valid date.", None
    if time_slot not in TIME_SLOTS:
        return False, "Please select a valid time slot.", None
    if consultation_type not in CONSULTATION_TYPES:
        return False, "Please select a valid consultation type.", None
    if status not in APPOINTMENT_STATUSES:
        return False, "Invalid appointment status.", None

    try:
        with db_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO appointments (user_email, date, time_slot, consultation_type, notes, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_email, appt_date, time_slot, consultation_type, notes, status),
            )
            appt_id = int(cur.lastrowid)
        return True, "Your appointment has been successfully scheduled.", appt_id
    except Exception:
        return False, "Could not book appointment. Please try again.", None


def fetch_appointments(user_email: str) -> pd.DataFrame:
    """Fetch all appointments for a user."""

    user_email = (user_email or "").strip().lower()
    if not user_email:
        return pd.DataFrame(columns=["id", "user_email", "date", "time_slot", "consultation_type", "notes", "status"])

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, user_email, date, time_slot, consultation_type, notes, status
            FROM appointments
            WHERE user_email = ?
            ORDER BY
                date ASC,
                CASE time_slot
                    WHEN 'Morning' THEN 0
                    WHEN 'Afternoon' THEN 1
                    WHEN 'Evening' THEN 2
                    ELSE 3
                END ASC,
                id DESC
            """,
            (user_email,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["id", "user_email", "date", "time_slot", "consultation_type", "notes", "status"])

    return pd.DataFrame([dict(r) for r in rows])


def cancel_appointment(*, appointment_id: int, user_email: str) -> Tuple[bool, str]:
    """Cancel a scheduled appointment for the given user."""

    user_email = (user_email or "").strip().lower()
    try:
        with db_conn() as conn:
            cur = conn.execute(
                """
                UPDATE appointments
                SET status = 'Cancelled'
                WHERE id = ? AND user_email = ? AND status = 'Scheduled'
                """,
                (int(appointment_id), user_email),
            )
        if int(cur.rowcount or 0) == 0:
            return False, "Could not cancel: appointment not found or not scheduled."
        return True, "Appointment cancelled."
    except Exception:
        return False, "Could not cancel appointment. Please try again."


def complete_appointment(*, appointment_id: int, user_email: str) -> Tuple[bool, str]:
    """Mark a scheduled appointment as completed for the given user."""

    user_email = (user_email or "").strip().lower()
    try:
        with db_conn() as conn:
            cur = conn.execute(
                """
                UPDATE appointments
                SET status = 'Completed'
                WHERE id = ? AND user_email = ? AND status = 'Scheduled'
                """,
                (int(appointment_id), user_email),
            )
        if int(cur.rowcount or 0) == 0:
            return False, "Could not update: appointment not found or not scheduled."
        return True, "Appointment marked as completed."
    except Exception:
        return False, "Could not update appointment. Please try again."


# -------------------------------
# UI helpers (dark theme + cards)
# -------------------------------


def inject_css() -> None:
    """Inject a minimal, modern dark UI layer on top of Streamlit."""

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }

                    /* Hide Streamlit chrome for a more app-like feel */
                    #MainMenu { visibility: hidden; }
                    footer { visibility: hidden; }
                    header { visibility: hidden; }

                    /* Keep the sidebar hamburger visible for navigation */
                    [data-testid="collapsedControl"] {
                        visibility: visible;
                        pointer-events: auto;
                    }

          /* Cards */
          .nn-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px 16px;
          }
          .nn-card h4 { margin: 0 0 6px 0; font-weight: 600; font-size: 0.95rem; opacity: 0.85; }
          .nn-value { font-size: 2.2rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.0; }
          .nn-sub { margin-top: 6px; opacity: 0.75; font-size: 0.9rem; }

          .nn-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.04);
            font-weight: 650;
          }

                    .nn-auth-shell {
                        padding: 20px;
                        border-radius: 18px;
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        background: rgba(255, 255, 255, 0.03);
                    }
                    .nn-auth-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; }
                    .nn-auth-sub { opacity: 0.78; margin-top: 6px; }

          .nn-small-metric {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 12px 12px;
          }
          .nn-small-metric .k { opacity: 0.75; font-size: 0.9rem; }
          .nn-small-metric .v { font-size: 1.25rem; font-weight: 700; }

          /* Reduce default table glare */
          [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

          /* Sidebar spacing */
          section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, *, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="nn-card">
          <h4>{title}</h4>
                    <div class="nn-value">{value}</div>
          <div class="nn-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, *, emoji: str) -> None:
    st.markdown(
        f"""
        <div class="nn-badge">
          <span style="font-size: 1.1rem">{emoji}</span>
          <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def small_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="nn-small-metric">
          <div class="k">{label}</div>
          <div class="v">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------
# Session state
# -------------------------------


def ensure_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_id": None,
        "user": None,
        "sim_state": None,
        "latest_reading": None,
        "recent_readings": [],
        "emergency_last_sent": None,
        "nav_public": "🔐 Login",
        "nav_private": "🏠 Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_logged_in(user: sqlite3.Row) -> None:
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = int(user["id"])
    st.session_state["user"] = dict(user)
    # Default landing page after login.
    st.session_state["nav_private"] = "🏠 Dashboard"


def logout() -> None:
    for k in [
        "authenticated",
        "user_id",
        "user",
        "sim_state",
        "latest_reading",
        "recent_readings",
        "emergency_last_sent",
        "nav_public",
        "nav_private",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    ensure_session_state()


# -------------------------------
# Pages
# -------------------------------


PSYCHIATRIST_DIRECTORY: List[Dict[str, str]] = [
    {
        "Name": "Dr. Aisha Rao, MD",
        "Specialty": "Psychiatrist",
        "Focus": "Stress, anxiety, burnout",
        "Mode": "Online / Clinic",
        "Availability": "Today 6:00 PM",
        "Contact": "clinic+demo@neuranexus.local",
    },
    {
        "Name": "Dr. Kunal Mehta, MD",
        "Specialty": "Psychiatrist",
        "Focus": "Sleep & mood support",
        "Mode": "Online",
        "Availability": "Tomorrow 10:30 AM",
        "Contact": "+91-90000-00002",
    },
    {
        "Name": "Dr. Sara Fernandes, MD",
        "Specialty": "Psychiatrist",
        "Focus": "Work stress & panic symptoms",
        "Mode": "Clinic",
        "Availability": "Tomorrow 4:15 PM",
        "Contact": "+91-90000-00003",
    },
]


def _remember_reading(reading: Dict[str, Any], *, keep: int = 20) -> None:
    st.session_state["latest_reading"] = reading
    recent: List[Dict[str, Any]] = st.session_state.get("recent_readings", [])
    recent.append(
        {
            "timestamp": reading["timestamp"],
            "alpha": reading["alpha"],
            "beta": reading["beta"],
            "theta": reading["theta"],
            "stress_index": reading["stress_index"],
            "status": reading["status"],
            "severity": reading["severity"],
        }
    )
    st.session_state["recent_readings"] = recent[-keep:]


def _render_risk_banner(reading: Dict[str, Any]) -> None:
    severity = str(reading.get("severity", ""))
    if severity == "High":
        _toast("High risk detected", icon="⚠️")
        st.error("High risk detected. Consider taking a break and reaching out for support.")
    elif severity == "Moderate":
        st.warning("Elevated stress detected. Try a short breathing exercise.")
    else:
        st.success("Risk level normal.")


def _render_auto_consulting(reading: Dict[str, Any]) -> None:
    if str(reading.get("severity")) != "High":
        return

    st.markdown("### 🧑‍⚕️ Auto Consulting (Recommended)")
    st.caption("Dummy professional directory for demo purposes.")

    for pro in PSYCHIATRIST_DIRECTORY:
        with st.container(border=True):
            st.markdown(f"**{pro['Name']}** — {pro['Specialty']}")
            st.caption(f"Focus: {pro['Focus']} • Mode: {pro['Mode']} • Availability: {pro['Availability']}")
            st.write(f"Contact: {pro['Contact']}")


def _get_latest_reading_from_db(user_id: int) -> Optional[Dict[str, Any]]:
    df = fetch_history(user_id, limit=1)
    if df.empty:
        return None

    row = df.iloc[-1]
    badge_emoji = classify_stress(float(row["stress_index"]))[2]
    return {
        "timestamp": row["timestamp"].to_pydatetime().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "theta": float(row["theta"]),
        "stress_index": float(row["stress_index"]),
        "status": str(row["status"]),
        "severity": str(row["severity"]),
        "badge": badge_emoji,
    }


def page_login() -> None:
    st.markdown(
        f"""
        <div class="nn-auth-shell">
          <div class="nn-auth-title">🧠 {APP_TITLE}</div>
                    <div class="nn-auth-sub">Sign in to your account</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([1.0, 1.1, 1.0])
    with center:
        with st.container(border=True):
            st.markdown("### Sign in")
            with st.form("login_form", border=False):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", **_wide_kwargs(st.form_submit_button))

            if submitted:
                ok, msg, user = authenticate(email, password)
                if ok and user is not None:
                    set_logged_in(user)
                    _toast("Signed in", icon="✅")
                    _rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.caption("Don't have an account?")
            if st.button("Create account", **_wide_kwargs(st.button)):
                st.session_state["nav_public"] = "🆕 Signup"
                _rerun()


def page_signup() -> None:
    st.markdown(
        f"""
        <div class="nn-auth-shell">
          <div class="nn-auth-title">🧠 {APP_TITLE}</div>
                    <div class="nn-auth-sub">Create your account</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([1.0, 1.25, 1.0])
    with center:
        with st.container(border=True):
            st.markdown("### Signup")
            with st.form("signup_form", border=False):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                c1, c2 = st.columns(2)
                with c1:
                    password = st.text_input("Password", type="password", help="Minimum 8 characters")
                with c2:
                    confirm = st.text_input("Confirm Password", type="password")

                c3, c4 = st.columns(2)
                with c3:
                    age = st.number_input("Age", min_value=10, max_value=120, value=22, step=1)
                with c4:
                    emergency = st.text_input("Emergency Contact", placeholder="Name + phone/email")

                submitted = st.form_submit_button("Create Account", **_wide_kwargs(st.form_submit_button))

            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(
                        name=name,
                        email=email,
                        password=password,
                        age=int(age) if age else None,
                        emergency_contact=emergency,
                    )
                    if ok:
                        st.success(msg)
                        st.session_state["nav_public"] = "🔐 Login"
                        _rerun()
                    else:
                        st.error(msg)

            st.markdown("---")
            st.caption("Already have an account?")
            if st.button("Back to sign in", **_wide_kwargs(st.button)):
                st.session_state["nav_public"] = "🔐 Login"
                _rerun()


def page_dashboard() -> None:
    user = st.session_state.get("user") or {}
    name = user.get("name", "User")
    user_id = int(st.session_state["user_id"])

    st.markdown("## 🏠 Dashboard")
    st.caption(f"Hello, **{name}**")

    # Manual simulation controls (no auto-refresh).
    with st.container(border=True):
        st.markdown("### 🎛️ Generate Reading")
        c1, c2 = st.columns([1.35, 0.65])
        with c1:
            intensity = st.slider(
                "Simulation intensity",
                min_value=0,
                max_value=100,
                value=35,
                key="dash_intensity",
                help="Higher intensity increases beta relative to alpha.",
            )
            save_to_history = st.checkbox("Save to History", value=True, key="dash_save_history")
        with c2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            generate = st.button("Generate Reading", key="dash_generate", **_wide_kwargs(st.button))
            st.caption("Manual simulation")

    reading = st.session_state.get("latest_reading") or _get_latest_reading_from_db(user_id)

    if generate:
        reading = generate_reading(stress_level=float(intensity) / 100.0)
        _remember_reading(reading)
        if save_to_history:
            log_reading(user_id, reading)
            _toast("Saved to history", icon="💾")
        else:
            _toast("Reading generated", icon="🧠")

    if reading is None:
        st.info("Generate your first reading above to unlock insights and recommendations.")
        return

    # Top metrics
    c1, c2, c3, c4 = st.columns([1.25, 1.0, 1.0, 1.2])
    with c1:
        metric_card("Stress Index", f"{float(reading['stress_index']):.2f}", sub="beta / alpha")
    with c2:
        metric_card("Status", f"{reading['status']}")
    with c3:
        metric_card("Severity", f"{reading['severity']}")
    with c4:
        metric_card("Last Update", str(reading["timestamp"]).replace("T", " ").replace("+00:00", " UTC"))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    badge(f"{reading['status']} • {reading['severity']}", emoji=reading.get("badge", "🧠"))

    _render_risk_banner(reading)

    # Smart integration: recommend booking a consultation when stress is HIGH.
    if str(reading.get("status")) == "HIGH":
        urgent = float(reading.get("stress_index", 0.0)) > STRESS_THRESHOLD_SEVERE
        with st.container(border=True):
            if urgent:
                st.error("Urgent consultation recommended (stress index > 2).")
                st.caption("Emergency Support is recommended.")
            else:
                st.warning("We recommend booking a consultation.")

            c1, c2 = st.columns([1.0, 0.35])
            with c1:
                st.write("Schedule a consultation to get guided support.")
            with c2:
                if st.button("👉 Book Now", key="dash_book_now", **_wide_kwargs(st.button)):
                    st.session_state["nav_private"] = "📅 Book Appointment"
                    _rerun()

    band_cols = st.columns(3)
    with band_cols[0]:
        small_metric("Theta (4–8 Hz)", f"{float(reading['theta']):.3f}")
    with band_cols[1]:
        small_metric("Alpha (8–12 Hz)", f"{float(reading['alpha']):.3f}")
    with band_cols[2]:
        small_metric("Beta (13–30 Hz)", f"{float(reading['beta']):.3f}")

    # Auto consulting suggestions when HIGH
    _render_auto_consulting(reading)

    st.markdown("### 🧾 Recent Readings")
    df = fetch_history(user_id, limit=30)
    if df.empty:
        st.caption("No records in history yet.")
        return

    show = df.copy()
    show["timestamp"] = show["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    show = show.tail(30)
    st.dataframe(show, hide_index=True, **_wide_kwargs(st.dataframe))


def page_simulation() -> None:
    st.markdown("## 🎛️ Manual Simulation")
    st.caption("Choose an intensity level and generate a reading. No auto-refresh.")

    user_id = int(st.session_state["user_id"])

    with st.container(border=True):
        intensity = st.slider(
            "Simulation intensity",
            min_value=0,
            max_value=100,
            value=35,
            help="Higher intensity increases beta relative to alpha.",
        )
        save_to_history = st.checkbox("Save this reading to History", value=True)
        generate = st.button("Generate Reading", **_wide_kwargs(st.button))

    if generate:
        reading = generate_reading(stress_level=float(intensity) / 100.0)
        _remember_reading(reading)

        if save_to_history:
            log_reading(user_id, reading)
            _toast("Saved to history", icon="💾")

        st.markdown("### ✅ Result")
        c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
        with c1:
            metric_card("Stress Index", f"{reading['stress_index']:.2f}", sub="beta / alpha")
        with c2:
            metric_card("Status", reading["status"])
        with c3:
            metric_card("Severity", reading["severity"])

        badge(f"{reading['status']} • {reading['severity']}", emoji=reading["badge"])
        _render_risk_banner(reading)

        band_cols = st.columns(3)
        with band_cols[0]:
            small_metric("Theta (4–8 Hz)", f"{reading['theta']:.3f}")
        with band_cols[1]:
            small_metric("Alpha (8–12 Hz)", f"{reading['alpha']:.3f}")
        with band_cols[2]:
            small_metric("Beta (13–30 Hz)", f"{reading['beta']:.3f}")

        _render_auto_consulting(reading)
    else:
        last = st.session_state.get("latest_reading")
        if last is not None:
            st.caption("Last generated reading is available on the Dashboard.")


def page_consulting() -> None:
    st.markdown("## 🧑‍⚕️ Consulting")
    st.caption("Automatic suggestions appear when risk is HIGH (dummy data).")

    user_id = int(st.session_state["user_id"])
    reading = st.session_state.get("latest_reading") or _get_latest_reading_from_db(user_id)

    if reading is None:
        st.info("Generate a reading first (Manual Simulation).")
        return

    badge(f"Current: {reading['status']} • {reading['severity']}", emoji=reading.get("badge", "🧠"))

    if str(reading.get("severity")) == "High":
        st.success("Auto-consulting is enabled for HIGH risk.")
        _render_auto_consulting(reading)
    else:
        st.info("No HIGH-risk reading detected. You can still book a consultation (simulated).")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 📅 Book Consultation (Simulated)")
        st.write("This is a local demo — no external APIs are used.")
        if st.button("Request Consultation", **_wide_kwargs(st.button)):
            st.success("Consultation request submitted (simulated).")


def page_emergency() -> None:
    st.markdown("## 🚨 Emergency")
    st.caption("Simulated emergency contact workflow (no real SMS/call).")

    user = st.session_state.get("user") or {}
    contact = user.get("emergency_contact", "")

    with st.container(border=True):
        st.markdown("### Saved Emergency Contact")
        st.write(contact if contact else "(Not set)")

        if st.button("Send Emergency Alert (Simulated)", **_wide_kwargs(st.button)):
            st.session_state["emergency_last_sent"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            st.success(f"Simulated alert sent to: **{contact}**")

        last_sent = st.session_state.get("emergency_last_sent")
        if last_sent:
            st.caption(f"Last simulated alert: {last_sent}")


def page_history() -> None:
    st.markdown("## 🕒 History")
    st.caption("Stored locally in SQLite.")

    user_id = int(st.session_state["user_id"])
    df = fetch_history(user_id, limit=2500)

    if df.empty:
        st.info("No history yet. Generate entries from the Dashboard.")
        return

    last_row = df.iloc[-1]
    badge_emoji = classify_stress(float(last_row["stress_index"]))[2]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Latest Stress Index", f"{float(last_row['stress_index']):.2f}")
    with c2:
        metric_card("Total Records", f"{len(df):,}")
    with c3:
        badge(f"{last_row['status']} • {last_row['severity']}", emoji=badge_emoji)

    show = df.copy()
    show["timestamp"] = show["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    show = show.tail(400)
    st.dataframe(show, hide_index=True, **_wide_kwargs(st.dataframe))


def page_book_appointment() -> None:
    st.markdown("## 📅 Book Appointment")
    st.caption("Schedule a simulated consultation (stored locally in SQLite).")

    user = st.session_state.get("user") or {}
    name = str(user.get("name", ""))
    email = str(user.get("email", "")).strip().lower()

    user_id = int(st.session_state["user_id"])
    reading = st.session_state.get("latest_reading") or _get_latest_reading_from_db(user_id)

    urgent = False
    suggested_type = "Mental Wellness"
    if reading is not None and str(reading.get("status")) == "HIGH":
        if float(reading.get("stress_index", 0.0)) > STRESS_THRESHOLD_SEVERE:
            urgent = True
            suggested_type = "Emergency Support"
            st.error("Urgent consultation recommended based on your latest reading.")
        else:
            suggested_type = "Stress Counseling"
            st.warning("We recommend booking a consultation based on your latest reading.")

    if not email:
        st.error("Missing user email. Please log in again.")
        return

    default_type_index = int(CONSULTATION_TYPES.index(suggested_type)) if suggested_type in CONSULTATION_TYPES else 1

    with st.container(border=True):
        with st.form("book_appt_form", border=False):
            st.text_input("User Name", value=name, disabled=True)
            st.text_input("Email", value=email, disabled=True)

            c1, c2 = st.columns(2)
            with c1:
                appt_dt = st.date_input("Select Date", value=date.today())
            with c2:
                time_slot = st.selectbox("Select Time Slot", list(TIME_SLOTS), index=0)

            consult_type = st.selectbox(
                "Consultation Type",
                list(CONSULTATION_TYPES),
                index=default_type_index,
                help="Emergency Support is recommended when stress index is very high.",
            )
            notes = st.text_area("Notes (optional)", height=90, placeholder="Any context you'd like to share...")
            submitted = st.form_submit_button("Book Appointment", **_wide_kwargs(st.form_submit_button))

    if submitted:
        if appt_dt < date.today():
            st.error("Please choose a future date (or today).")
            return

        if urgent and consult_type != "Emergency Support":
            st.warning("Emergency Support is recommended for urgent cases.")

        ok, msg, appt_id = create_appointment(
            user_email=email,
            appt_date=appt_dt.isoformat(),
            time_slot=time_slot,
            consultation_type=consult_type,
            notes=notes,
            status="Scheduled",
        )

        if not ok:
            st.error(msg)
            return

        st.success(msg)
        with st.container(border=True):
            st.markdown("### ✅ Appointment Summary")
            st.write(f"**Appointment ID:** #{appt_id}")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Date:** {appt_dt.isoformat()}")
            st.write(f"**Time Slot:** {time_slot}")
            st.write(f"**Type:** {consult_type}")
            st.write("**Status:** Scheduled")
            if notes:
                st.write(f"**Notes:** {notes}")


def page_my_appointments() -> None:
    st.markdown("## 📋 My Appointments")
    st.caption("Manage your scheduled consultations.")

    user = st.session_state.get("user") or {}
    email = str(user.get("email", "")).strip().lower()
    if not email:
        st.error("Missing user email. Please log in again.")
        return

    df = fetch_appointments(email)
    if df.empty:
        st.info("No appointments yet. Use **Book Appointment** to schedule one.")
        return

    show = df[["id", "date", "time_slot", "consultation_type", "status"]].copy()
    show = show.rename(columns={"consultation_type": "type"})
    st.dataframe(show, hide_index=True, **_wide_kwargs(st.dataframe))

    options: Dict[str, int] = {}
    for _, row in show.iterrows():
        label = f"#{int(row['id'])} • {row['date']} • {row['time_slot']} • {row['type']} • {row['status']}"
        options[label] = int(row["id"])

    with st.container(border=True):
        st.markdown("### ⚙️ Manage Appointment")
        selected_label = st.selectbox("Select an appointment", list(options.keys()))
        appt_id = int(options[selected_label])

        current_status = str(df.loc[df["id"] == appt_id, "status"].iloc[0])

        c1, c2 = st.columns(2)
        with c1:
            cancel = st.button(
                "Cancel appointment",
                key="appt_cancel",
                disabled=current_status != "Scheduled",
                **_wide_kwargs(st.button),
            )
        with c2:
            complete = st.button(
                "Mark as completed",
                key="appt_complete",
                disabled=current_status != "Scheduled",
                **_wide_kwargs(st.button),
            )

        if cancel:
            ok, msg = cancel_appointment(appointment_id=appt_id, user_email=email)
            (st.success if ok else st.error)(msg)
            _rerun()

        if complete:
            ok, msg = complete_appointment(appointment_id=appt_id, user_email=email)
            (st.success if ok else st.error)(msg)
            _rerun()


def page_support() -> None:
    st.markdown("## 🤝 Support")
    st.caption("Resources and emergency tools (simulated).")

    user = st.session_state.get("user") or {}
    user_id = int(st.session_state["user_id"])
    reading = st.session_state.get("latest_reading") or _get_latest_reading_from_db(user_id)

    if reading is not None and str(reading.get("status")) == "HIGH":
        urgent = float(reading.get("stress_index", 0.0)) > STRESS_THRESHOLD_SEVERE
        if urgent:
            st.error("Urgent consultation recommended.")
        else:
            st.warning("Elevated stress detected. Consider booking a consultation.")
        if st.button("👉 Book Now", key="support_book_now", **_wide_kwargs(st.button)):
            st.session_state["nav_private"] = "📅 Book Appointment"
            _rerun()

    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        st.markdown(
            """
            <div class="nn-card">
              <h4>🌿 Mental Wellness Tips</h4>
              <div class="nn-sub">Simple habits can reduce daily stress.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.write("• 60-second breathing reset between tasks")
        st.write("• Short walk + hydration break")
        st.write("• Reduce caffeine late in the day")
        st.write("• Keep sleep schedule consistent")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="nn-card">
              <h4>🧭 When to Seek Help</h4>
              <div class="nn-sub">Reach out if symptoms persist or worsen.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.write("• Persistent anxiety / low mood")
        st.write("• Sleep disruption affecting daily function")
        st.write("• Difficulty coping with responsibilities")
        st.write("• Thoughts of self-harm — seek immediate help")

    with c2:
        contact = str(user.get("emergency_contact", "")).strip()
        with st.container(border=True):
            st.markdown("### 🚨 Emergency Contact")
            st.write(contact if contact else "(Not set)")

            if st.button("Send Emergency Alert (Simulated)", key="support_emergency", **_wide_kwargs(st.button)):
                st.session_state["emergency_last_sent"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                st.success(f"Simulated alert sent to: **{contact}**")

            last_sent = st.session_state.get("emergency_last_sent")
            if last_sent:
                st.caption(f"Last simulated alert: {last_sent}")

        if reading is not None:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            badge(f"Current: {reading['status']} • {reading['severity']}", emoji=reading.get("badge", "🧠"))


def page_resources() -> None:
    # Backwards-compatible alias (Support is the main destination in navigation).
    page_support()


def page_resources() -> None:
    st.markdown("## 📚 Resources")
    st.caption("General wellness tips (not medical advice).")

    c1, c2 = st.columns([1.2, 1.0])

    with c1:
        st.markdown(
            """
            <div class="nn-card">
              <h4>🌿 Mental Wellness Tips</h4>
              <div class="nn-sub">Simple habits can reduce daily stress.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.write("• 60-second breathing reset between tasks")
        st.write("• Short walk + hydration break")
        st.write("• Reduce caffeine late in the day")
        st.write("• Keep sleep schedule consistent")

    with c2:
        st.markdown(
            """
            <div class="nn-card">
              <h4>🧭 When to Seek Help</h4>
              <div class="nn-sub">Reach out if symptoms persist or worsen.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.write("• Persistent anxiety / low mood")
        st.write("• Sleep disruption affecting daily function")
        st.write("• Difficulty coping with responsibilities")
        st.write("• Thoughts of self-harm — seek immediate help")


def top_hamburger_nav(current_page: str) -> None:
    """Render an in-app top hamburger navigation control.

    This ensures navigation is visible even when Streamlit's built-in header
    chrome is hidden for a more app-like UI.
    """

    if not st.session_state.get("authenticated"):
        return

    nav_items = [
        "🏠 Dashboard",
        "🕒 History",
        "📅 Book Appointment",
        "📋 My Appointments",
        "🤝 Support",
    ]

    user = st.session_state.get("user") or {}

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.12, 0.62, 0.26])

        with c1:
            if hasattr(st, "popover"):
                with st.popover("☰"):
                    st.caption("Navigation")
                    for i, item in enumerate(nav_items):
                        if st.button(item, key=f"topnav_item_{i}", **_wide_kwargs(st.button)):
                            st.session_state["nav_private"] = item
                            _rerun()
                    st.markdown("---")
                    if st.button("🚪 Logout", key="topnav_logout", **_wide_kwargs(st.button)):
                        logout()
                        _rerun()
            else:  # pragma: no cover
                with st.expander("☰ Menu", expanded=False):
                    for i, item in enumerate(nav_items):
                        if st.button(item, key=f"topnav_item_{i}", **_wide_kwargs(st.button)):
                            st.session_state["nav_private"] = item
                            _rerun()
                    st.markdown("---")
                    if st.button("🚪 Logout", key="topnav_logout", **_wide_kwargs(st.button)):
                        logout()
                        _rerun()

        with c2:
            st.markdown(f"**{APP_TITLE}**")
            st.caption(str(current_page))

        with c3:
            st.markdown(f"**{user.get('name','')}**")
            st.caption(user.get("email", ""))


def sidebar_nav() -> str:
    if not st.session_state.get("authenticated"):
        return str(st.session_state.get("nav_public", "🔐 Login"))

    # Authenticated: sidebar navigation.
    nav_items = [
        "🏠 Dashboard",
        "🕒 History",
        "📅 Book Appointment",
        "📋 My Appointments",
        "🤝 Support",
    ]

    current = str(st.session_state.get("nav_private", nav_items[0]))
    if current not in nav_items:
        st.session_state["nav_private"] = nav_items[0]

    st.sidebar.markdown(f"# 🧠 {APP_TITLE}")
    st.sidebar.caption(APP_TAGLINE)
    st.sidebar.markdown("---")

    user = st.session_state.get("user") or {}
    st.sidebar.markdown(f"**Signed in**\n\n{user.get('name','')}")
    st.sidebar.caption(user.get("email", ""))
    st.sidebar.markdown("---")

    current = st.sidebar.radio("Navigate", nav_items, key="nav_private")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", **_wide_kwargs(st.sidebar.button)):
        logout()
        _rerun()

    return str(current)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

    init_db()
    ensure_session_state()
    inject_css()

    if not st.session_state.get("authenticated"):
        # Public auth screens should look like a real login page (no sidebar navigation).
        st.markdown(
            '<style>section[data-testid="stSidebar"]{display:none;}[data-testid="collapsedControl"]{display:none;}</style>',
            unsafe_allow_html=True,
        )
        page = sidebar_nav()
        if page == "🆕 Signup":
            page_signup()
        else:
            page_login()
        return

    page = sidebar_nav()
    top_hamburger_nav(page)

    if page == "🏠 Dashboard":
        page_dashboard()
        return
    if page == "🕒 History":
        page_history()
        return
    if page == "📅 Book Appointment":
        page_book_appointment()
        return
    if page == "📋 My Appointments":
        page_my_appointments()
        return
    if page == "🤝 Support":
        page_support()
        return

    page_dashboard()


main()
