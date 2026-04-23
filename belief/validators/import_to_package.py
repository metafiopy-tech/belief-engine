"""Import-name → PyPI package-name translation table — Session 3 (v3.2).

Python's ``import`` name is not always the same as the pip package
name.  ``import cv2`` is installed via ``pip install opencv-python``;
``import sklearn`` via ``pip install scikit-learn``; ``import jwt``
via ``pip install PyJWT``.  Getting this wrong is a subtle security
risk: if the LLM emits ``pip install sklearn`` (thinking that's the
package), it actually resolves to a legitimate-but-different package
(``sklearn`` is a deprecated meta-package) and obscures intent.

This table is hand-curated.  **Do NOT use pipreqs's table** — its
fallback behaviour is to return the import name unchanged when no
mapping exists, which caused a real dependency-confusion CVE at
LSGeurope in 2023.  Explicit mappings only; unknown names return
unchanged.

Public API::

    resolve_import_to_package("cv2") → "opencv-python"
    resolve_import_to_package("httpx") → "httpx"     # no mapping, passthrough
    resolve_import_to_package("") → ""
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Curated mappings.  Ordered to keep related families grouped for grep-ability.
# ---------------------------------------------------------------------------

_IMPORT_TO_PACKAGE: Final[dict[str, str]] = {
    # Computer vision / imaging
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "fitz": "PyMuPDF",            # PyMuPDF docs routinely use "import fitz"

    # Machine learning / data
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",

    # Web / scraping
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",

    # Auth / crypto
    "jwt": "PyJWT",
    "Crypto": "pycryptodome",

    # Databases
    "MySQLdb": "mysqlclient",

    # Docs & office
    "docx": "python-docx",
    "pptx": "python-pptx",
    "xlrd": "xlrd",               # same, listed for completeness
    "openpyxl": "openpyxl",       # same, listed so grep finds it

    # Messaging / RPC
    "grpc": "grpcio",
    "zmq": "pyzmq",

    # Community APIs with quirky package names
    "discord": "discord.py",
    "google.generativeai": "google-generativeai",

    # Attrs — the import is ``attr`` (historical), package is ``attrs``
    "attr": "attrs",

    # Qt binding libs (identity mappings but common source of confusion —
    # grep-find-friendly)
    "PyQt5": "PyQt5",
    "PyQt6": "PyQt6",
    "PySide2": "PySide2",
    "PySide6": "PySide6",
}


def resolve_import_to_package(import_name: str) -> str:
    """Return the PyPI package name for an import name, or the input
    unchanged if there's no known mapping.

    The function is case-sensitive — ``PIL`` maps (capital), ``pil``
    does not.  This matches how Python itself spells imports.

    Dotted imports are checked as a whole first (``google.generativeai``),
    then the top-level module (``google``) if no full match.  The
    top-level fallback is intentionally NOT a transform that picks up
    arbitrary submodules — ``google.cloud.storage`` would return
    itself, not some wrong package, because we'd rather leave it to
    the validator's fuzzy-match layer than guess.
    """
    if not import_name:
        return ""
    if import_name in _IMPORT_TO_PACKAGE:
        return _IMPORT_TO_PACKAGE[import_name]
    top = import_name.split(".", 1)[0]
    if top in _IMPORT_TO_PACKAGE:
        return _IMPORT_TO_PACKAGE[top]
    return import_name


def all_known_mappings() -> dict[str, str]:
    """Return a copy of the full mapping table — useful for CLI tools
    that want to list what we cover (``belief validator list-mappings``).
    """
    return dict(_IMPORT_TO_PACKAGE)


__all__ = [
    "all_known_mappings",
    "resolve_import_to_package",
]
