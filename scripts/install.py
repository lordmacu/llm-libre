#!/usr/bin/env python3
"""Entry point for the installer: `python3 scripts/install.py`.

A thin shim on purpose. The logic lives in `src/llm_libre/installer.py` so it
can be imported and tested; this file only makes `src/` importable when the
package has not been installed yet -- which, in an installer, is always.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_libre.installer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
