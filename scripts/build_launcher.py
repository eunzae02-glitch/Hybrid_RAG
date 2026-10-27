"""Optionally rebuild the Windows launcher using the current Python environment."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable, "-X", "utf8", "-m", "PyInstaller",
            "--noconfirm", "--clean", "--onefile", "--windowed",
            "--name", "논문RAG 실행", "--distpath", str(ROOT),
            "--workpath", str(ROOT / ".local" / "launcher-build"),
            "--specpath", str(ROOT / ".local"),
            "--exclude-module", "torch", "--exclude-module", "transformers",
            "--exclude-module", "numpy", "--exclude-module", "chromadb",
            "--exclude-module", "streamlit", "--exclude-module", "PIL",
            "--exclude-module", "google", str(ROOT / "launcher.py"),
        ],
        cwd=ROOT,
        check=True,
    )
