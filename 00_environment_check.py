from __future__ import annotations

import importlib
import platform
import sys

from project_utils import ensure_output_dirs, get_paths

PACKAGES = ["pandas", "numpy", "matplotlib", "sklearn", "torch", "tqdm"]


def main() -> None:
    ensure_output_dirs()
    print("=== Environment check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    paths = get_paths()
    print(f"Data directory: {paths['data_dir']}")
    print("\nPackage status:")
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"  [OK] {package} ({version})")
        except Exception:
            print(f"  [MISSING] {package}")
    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()
