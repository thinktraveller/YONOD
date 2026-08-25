from __future__ import annotations

from importlib import metadata
from pathlib import Path
import json
import sys


PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "rdkit": "rdkit",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "pyarrow": "pyarrow",
    "pytest": "pytest",
    "lightgbm": "lightgbm",
    "dscribe": "dscribe",
    "ase": "ase",
}


def package_status(distribution: str, import_name: str) -> dict:
    try:
        __import__(import_name)
        return {"installed": True, "version": metadata.version(distribution)}
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.paths import REPRO_ROOT, YONOD_ROOT, ensure_yonod_root

    result = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repro_root": str(REPRO_ROOT),
        "yonod_root": str(YONOD_ROOT),
        "yonod_package_exists": (ensure_yonod_root() / "yonod").exists(),
        "packages": {
            name: package_status(dist, import_name)
            for dist, import_name in PACKAGES.items()
            for name in [dist]
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    required = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "rdkit",
        "matplotlib",
        "seaborn",
        "PyYAML",
        "joblib",
        "tqdm",
        "pyarrow",
        "pytest",
    ]
    missing = [name for name in required if not result["packages"][name]["installed"]]
    return 1 if missing or not result["yonod_package_exists"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

