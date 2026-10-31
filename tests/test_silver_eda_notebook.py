"""Validate the generated notebook without notebook/data dependencies."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/04_generate_silver_eda_notebook.py"
NOTEBOOK = ROOT / "notebooks/v1/eda/01_silver_eda.ipynb"


def test_generated_notebook_is_current(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "silver.ipynb"
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)], check=True)
    assert output.read_text(encoding="utf-8") == NOTEBOOK.read_text(encoding="utf-8")
    output.write_text("outdated", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--check",
                             "--output", str(output)], capture_output=True)
    assert result.returncode != 0


def test_notebook_syntax_and_charts():
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert doc["nbformat"] == 4
    ids = [cell["id"] for cell in doc["cells"]]
    assert len(ids) == len(set(ids))
    code_cells = [cell for cell in doc["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 8
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse(cell["source"])
    source = "\n".join(cell["source"] for cell in code_cells)
    for token in ("SilverReader", "silver.verify()", "iter_records()", "plt.show()",
                  "ax.imshow", "ax.boxplot", "TfidfVectorizer", "EXPORT_RESULTS = False"):
        assert token in source
