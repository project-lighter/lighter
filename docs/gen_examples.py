"""Render the canonical research README and its public results in the local site."""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).resolve().parents[1]
project = root / "projects/experiment_comparison"
page = "examples/compare-and-continue.md"
text = (project / "README.md").read_text()
links = {
    "../../docs/guides/compatibility.md": "../guides/compatibility.md",
    "(results.json)": "(comparison-results.json)",
}
for source, target in links.items():
    if source not in text:
        raise ValueError(f"Review the canonical example's changed link: {source}")
    text = text.replace(source, target)
with mkdocs_gen_files.open(page, "w") as stream:
    stream.write(text)
mkdocs_gen_files.set_edit_path(page, "../projects/experiment_comparison/README.md")
with mkdocs_gen_files.open("examples/comparison-results.json", "w") as stream:
    stream.write((project / "results.json").read_text())
