from pathlib import Path
import subprocess
import sys


def test_manuscript_release_contract():
    script = Path(__file__).resolve().parents[1] / "scripts/validate_manuscript.py"
    subprocess.run([sys.executable, str(script)], check=True)
