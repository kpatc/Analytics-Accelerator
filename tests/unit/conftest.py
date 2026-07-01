"""Unit-test conftest: ensure src/ is on sys.path so tests can import bcgx directly.

This allows tests to write `from bcgx.data.generator import ...` rather than
the longer `from src.bcgx.data.generator import ...` form.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add <repo-root>/src to sys.path so `import bcgx` works without pip install
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
