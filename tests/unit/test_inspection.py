"""Inspection reads source and records without evaluating their contents."""

import json

from lighter.engine.inspection import inspection_cli


def test_inspection_does_not_execute_imports_expressions_or_targets(tmp_path, capsys):
    config = tmp_path / "recipe.yaml"
    config.write_text("_imports_: [does_not_exist]\nmodel:\n  _target_: does_not_exist.Model\nvalue: '$1 / 0'\nseed: 11\n")
    assert inspection_cli(["inspect", str(config), "--json", "seed=42"])
    source = json.loads(capsys.readouterr().out)
    assert source == {
        "_imports_": ["does_not_exist"],
        "model": {"_target_": "does_not_exist.Model"},
        "value": "$1 / 0",
        "seed": 42,
    }


def test_record_inspection_list_show_and_diff(tmp_path, capsys):
    first = {"schema_version": 1, "attempt_id": "first", "stage": "fit", "status": "completed", "seed": 11}
    second = {**first, "attempt_id": "second", "seed": 42}
    for value in (first, second):
        directory = tmp_path / value["attempt_id"]
        directory.mkdir()
        (directory / "record.json").write_text(json.dumps(value))
    assert inspection_cli(["runs", "list", str(tmp_path), "--json"])
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert inspection_cli(["runs", "show", str(tmp_path / "first")])
    assert json.loads(capsys.readouterr().out) == first
    assert inspection_cli(["runs", "diff", str(tmp_path / "first"), str(tmp_path / "second")])
    assert json.loads(capsys.readouterr().out)[0]["after"] == 42
    assert not inspection_cli(["fit", "config.yaml"])
