"""Environment-driven settings. Secrets come from a local, gitignored .env only."""
from __future__ import annotations

import os

from dotenv import load_dotenv


class MissingConfig(RuntimeError):
    """Raised with the NAMES of missing variables, never their values."""


def require(*names: str) -> dict[str, str]:
    """Return the named variables, or raise MissingConfig listing which are unset."""
    load_dotenv()
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        raise MissingConfig(
            "Missing environment variables: " + ", ".join(missing) + " (see .env.example)"
        )
    return {n: os.environ[n].strip() for n in names}


def optional(name: str, default: str) -> str:
    load_dotenv()
    return os.environ.get(name, "").strip() or default
