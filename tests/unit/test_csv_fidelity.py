"""CSV publication preserves literal fields and never publishes a partial merge."""

import csv
from unittest.mock import patch

import pytest

from lighter.callbacks.csv_writer import CsvWriter
from lighter.utils.types.enums import Stage


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def test_csv_finalization_preserves_literal_fields(tmp_path, mock_trainer):
    rows = [
        ["0001", "NA"],
        ["900719925474099312345", "NULL"],
        ["", ""],
        ["comma,id", 'quoted "label"'],
        ["line\nbreak", "日本語"],
    ]
    path = tmp_path / "predictions.csv"
    writer = CsvWriter(path, ["id", "label"])
    writer.setup(mock_trainer, None, Stage.PREDICT)
    writer.write({"id": [row[0] for row in rows], "label": [row[1] for row in rows]}, None, 0, 0)
    writer.on_predict_epoch_end(mock_trainer, None)
    assert read_rows(path) == [["id", "label"], *rows]
    assert not list(tmp_path.glob("*.tmp*"))


def test_empty_csv_shard_publishes_header(tmp_path, mock_trainer):
    path = tmp_path / "empty.csv"
    writer = CsvWriter(path, ["id", "label"])
    writer.setup(mock_trainer, None, Stage.PREDICT)
    writer.on_predict_epoch_end(mock_trainer, None)
    assert read_rows(path) == [["id", "label"]]


@pytest.mark.parametrize("bad_rows", [[["wrong"], ["value"]], [["label", "id"], ["reordered", "fields"]]])
def test_invalid_shard_preserves_previous_publication(tmp_path, mock_trainer, bad_rows):
    path = tmp_path / "predictions.csv"
    path.write_text("previous complete output\n", encoding="utf-8")
    writer = CsvWriter(path, ["id", "label"])
    writer.setup(mock_trainer, None, Stage.PREDICT)
    writer.write({"id": ["0001"], "label": ["NA"]}, None, 0, 0)
    own_shard = writer._temp_path
    other_shard = tmp_path / "other-rank.csv"
    with other_shard.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(bad_rows)

    def gather(paths, local):
        paths[:] = [local, other_shard]

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_gather_object", side_effect=gather),
    ):
        with pytest.raises(ValueError, match="CSV shard"):
            writer.on_predict_epoch_end(mock_trainer, None)
    assert path.read_text(encoding="utf-8") == "previous complete output\n"
    assert own_shard.exists() and other_shard.exists()
    assert not list(tmp_path.glob(".predictions.csv.*"))


def test_long_text_is_transported_without_global_csv_policy_change(tmp_path, mock_trainer):
    original_limit = csv.field_size_limit()
    path = tmp_path / "long.csv"
    writer = CsvWriter(path, ["id", "text"])
    writer.setup(mock_trainer, None, Stage.PREDICT)
    writer.write({"id": ["0001"], "text": [('line,"quoted"\n日本語' * 20000)]}, None, 0, 0)
    writer._csv_file.flush()
    serialized = writer._temp_path.read_bytes()
    writer.on_predict_epoch_end(mock_trainer, None)
    assert path.read_bytes() == serialized
    assert csv.field_size_limit() == original_limit
