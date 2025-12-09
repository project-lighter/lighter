"""Unit tests for FileWriter and CsvWriter callbacks."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from lighter.callbacks.base_writer import BaseWriter
from lighter.callbacks.csv_writer import CsvWriter
from lighter.callbacks.file_writer import FileWriter, writer_registry
from lighter.model import LighterModule
from lighter.utils.types.enums import Stage


@pytest.fixture
def mock_system():
    """Create a mock LighterModule for testing."""
    return MagicMock(spec=LighterModule)


# =============================================================================
# BaseWriter Tests
# =============================================================================


class TestBaseWriter:
    """Test suite for BaseWriter callback."""

    def test_setup_warns_on_existing_path(self, tmp_path, mock_trainer, mock_system):
        """Test that setup warns when path already exists."""

        # Create a concrete implementation for testing
        class ConcreteWriter(BaseWriter):
            def write(self, outputs, batch, batch_idx, dataloader_idx):
                pass

        # Create existing file
        existing_file = tmp_path / "existing.csv"
        existing_file.touch()

        writer = ConcreteWriter(path=existing_file)

        # Should complete without error (just logs a warning)
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        assert writer.path == existing_file

    def test_on_predict_batch_end_with_outputs(self, tmp_path, mock_trainer, mock_system):
        """Test on_predict_batch_end calls write when outputs exist."""
        write_called = []

        class ConcreteWriter(BaseWriter):
            def write(self, outputs, batch, batch_idx, dataloader_idx):
                write_called.append((outputs, batch, batch_idx, dataloader_idx))

        writer = ConcreteWriter(path=tmp_path / "test.csv")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"pred": torch.tensor([1, 2, 3])}
        batch = {"input": torch.tensor([1])}

        writer.on_predict_batch_end(mock_trainer, mock_system, outputs, batch, 5, 0)

        assert len(write_called) == 1
        assert write_called[0][0] == outputs
        assert write_called[0][2] == 5  # batch_idx

    def test_on_predict_batch_end_empty_outputs(self, tmp_path, mock_trainer, mock_system):
        """Test on_predict_batch_end skips when outputs are empty."""
        write_called = []

        class ConcreteWriter(BaseWriter):
            def write(self, outputs, batch, batch_idx, dataloader_idx):
                write_called.append(True)

        writer = ConcreteWriter(path=tmp_path / "test.csv")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Empty outputs
        writer.on_predict_batch_end(mock_trainer, mock_system, {}, None, 0, 0)

        # Write should not be called
        assert len(write_called) == 0


# =============================================================================
# FileWriter Tests
# =============================================================================


class TestFileWriter:
    """Test suite for FileWriter callback."""

    def test_initialization(self, tmp_path):
        """Test FileWriter initialization with built-in writer."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")
        assert writer.path == tmp_path
        assert writer.value_key == "pred"
        assert writer.name_key is None
        assert callable(writer.writer_fn)

    def test_initialization_custom_writer(self, tmp_path):
        """Test FileWriter initialization with custom writer function."""

        def custom_writer(path, tensor):
            pass

        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn=custom_writer)
        assert writer.writer_fn == custom_writer

    def test_initialization_invalid_writer(self, tmp_path):
        """Test FileWriter raises error for invalid writer."""
        with pytest.raises(ValueError, match="Writer with name 'invalid' is not registered"):
            FileWriter(directory=tmp_path, value_key="pred", writer_fn="invalid")

    def test_initialization_non_callable(self, tmp_path):
        """Test FileWriter raises error for non-callable writer."""
        with pytest.raises(TypeError, match="writer_fn must be a string or a callable"):
            FileWriter(directory=tmp_path, value_key="pred", writer_fn=123)

    def test_setup(self, tmp_path, mock_trainer, mock_system):
        """Test FileWriter setup creates directory and initializes counter."""
        writer = FileWriter(directory=tmp_path / "predictions", value_key="pred", writer_fn="tensor")

        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        assert writer.path.exists()
        assert writer._counter == 0
        assert writer._step == 1

    def test_setup_distributed(self, tmp_path, mock_trainer, mock_system):
        """Test FileWriter setup in distributed setting."""
        mock_trainer.world_size = 4
        mock_trainer.global_rank = 2

        writer = FileWriter(directory=tmp_path / "predictions", value_key="pred", writer_fn="tensor")

        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        assert writer._counter == 2  # Starts at rank
        assert writer._step == 4  # Step size = world_size

    def test_setup_file_path_error(self, tmp_path, mock_trainer, mock_system):
        """Test FileWriter raises error if path is a file."""
        file_path = tmp_path / "file.pt"

        writer = FileWriter(directory=file_path, value_key="pred", writer_fn="tensor")

        with pytest.raises(ValueError, match="expects 'directory' to be a directory path"):
            writer.setup(mock_trainer, mock_system, Stage.PREDICT)

    def test_write_with_tensor_batch(self, tmp_path, mock_trainer, mock_system):
        """Test writing a batch of tensor predictions."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {
            "pred": torch.tensor([[1, 2], [3, 4], [5, 6]])  # 3 predictions
        }

        writer.write(outputs, None, 0, 0)

        # Check files were created
        assert (tmp_path / "0.pt").exists()
        assert (tmp_path / "1.pt").exists()
        assert (tmp_path / "2.pt").exists()

        # Verify content
        loaded = torch.load(tmp_path / "0.pt")
        assert torch.equal(loaded, torch.tensor([1, 2]))

    def test_write_with_custom_names(self, tmp_path, mock_trainer, mock_system):
        """Test writing with custom sample names."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor", name_key="sample_id")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"pred": torch.tensor([[1, 2], [3, 4]]), "sample_id": ["patient_001", "patient_002"]}

        writer.write(outputs, None, 0, 0)

        assert (tmp_path / "patient_001.pt").exists()
        assert (tmp_path / "patient_002.pt").exists()

    def test_write_length_mismatch(self, tmp_path, mock_trainer, mock_system):
        """Test error when predictions and names have different lengths."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor", name_key="sample_id")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {
            "pred": torch.tensor([[1, 2], [3, 4], [5, 6]]),  # 3 items
            "sample_id": ["id1", "id2"],  # 2 items
        }

        with pytest.raises(ValueError, match="Length mismatch"):
            writer.write(outputs, None, 0, 0)

    def test_write_missing_key(self, tmp_path, mock_trainer, mock_system):
        """Test error when required key is missing from outputs."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"other_key": torch.tensor([1, 2, 3])}

        with pytest.raises(KeyError, match="expected key 'pred'"):
            writer.write(outputs, None, 0, 0)

    def test_to_sequence_scalar_tensor(self):
        """Test _to_sequence handles scalar tensors."""
        outputs = {"pred": torch.tensor(5.0)}
        result = FileWriter._to_sequence(outputs, "pred")
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor(5.0))

    def test_to_sequence_list(self):
        """Test _to_sequence handles lists."""
        outputs = {"pred": [1, 2, 3, 4]}
        result = FileWriter._to_sequence(outputs, "pred")
        assert result == [1, 2, 3, 4]

    def test_to_sequence_tuple(self):
        """Test _to_sequence handles tuples."""
        outputs = {"pred": (1, 2, 3)}
        result = FileWriter._to_sequence(outputs, "pred")
        assert result == [1, 2, 3]

    def test_prepare_value_tensor(self):
        """Test _prepare_value moves tensor to CPU."""
        tensor = torch.tensor([1, 2, 3])
        result = FileWriter._prepare_value(tensor)
        assert result.device.type == "cpu"

    def test_prepare_name_scalar_tensor(self):
        """Test _prepare_name handles scalar tensor."""
        name = torch.tensor(42)
        result = FileWriter._prepare_name(name)
        assert result == 42

    def test_prepare_name_vector_tensor(self):
        """Test _prepare_name handles vector tensor."""
        name = torch.tensor([1, 2, 3])
        result = FileWriter._prepare_name(name)
        assert result == [1, 2, 3]

    def test_prepare_name_non_tensor(self):
        """Test _prepare_name handles non-tensor values."""
        result = FileWriter._prepare_name("sample_001")
        assert result == "sample_001"

    def test_prepare_value_non_tensor(self):
        """Test _prepare_value handles non-tensor values."""
        result = FileWriter._prepare_value("text_value")
        assert result == "text_value"

    def test_write_before_setup(self, tmp_path, mock_trainer, mock_system):
        """Test that write skips batch when called before setup."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")
        # Don't call setup - counter will be None

        outputs = {"pred": torch.tensor([[1, 2], [3, 4]])}

        # Should skip without error
        writer.write(outputs, None, 0, 0)

        # No files should be created
        assert len(list(tmp_path.glob("*.pt"))) == 0

    def test_write_empty_values(self, tmp_path, mock_trainer, mock_system):
        """Test write handles empty values gracefully."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Empty tensor
        outputs = {"pred": torch.tensor([])}

        # Should skip without error
        writer.write(outputs, None, 0, 0)

        # No files should be created
        assert len(list(tmp_path.glob("*.pt"))) == 0

    def test_to_sequence_generic_sequence(self):
        """Test _to_sequence handles generic sequences (not str/bytes)."""

        outputs = {"pred": range(5)}  # range is a Sequence
        result = FileWriter._to_sequence(outputs, "pred")
        assert result == [0, 1, 2, 3, 4]

    def test_to_sequence_single_value(self):
        """Test _to_sequence wraps single non-sequence value."""
        outputs = {"pred": 42}
        result = FileWriter._to_sequence(outputs, "pred")
        assert result == [42]

    def test_to_sequence_string_not_split(self):
        """Test _to_sequence doesn't split strings."""
        outputs = {"pred": "sample_001"}
        result = FileWriter._to_sequence(outputs, "pred")
        assert result == ["sample_001"]

    def test_write_with_nested_directory(self, tmp_path, mock_trainer, mock_system):
        """Test FileWriter creates nested directories when using custom names."""
        writer = FileWriter(directory=tmp_path / "outputs", value_key="pred", writer_fn="tensor", name_key="path")
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"pred": torch.tensor([[1, 2]]), "path": ["subfolder/sample"]}

        writer.write(outputs, None, 0, 0)

        assert (tmp_path / "outputs" / "subfolder" / "sample.pt").exists()


class TestWriterRegistry:
    """Test suite for writer registry."""

    def test_builtin_writers_exist(self):
        """Test that all built-in writers are registered."""
        assert "tensor" in writer_registry._registry
        assert "image_2d" in writer_registry._registry
        assert "image_3d" in writer_registry._registry
        assert "text" in writer_registry._registry

    def test_get_existing_writer(self):
        """Test getting an existing writer."""
        writer = writer_registry.get("tensor")
        assert callable(writer)

    def test_get_nonexistent_writer(self):
        """Test error when getting non-existent writer."""
        with pytest.raises(ValueError, match="Writer with name 'nonexistent' is not registered"):
            writer_registry.get("nonexistent")

    def test_tensor_writer(self, tmp_path):
        """Test tensor writer function."""
        from lighter.callbacks.file_writer import write_tensor

        path = tmp_path / "test"
        tensor = torch.tensor([1, 2, 3, 4])

        write_tensor(path, tensor)

        assert (tmp_path / "test.pt").exists()
        loaded = torch.load(tmp_path / "test.pt")
        assert torch.equal(loaded, tensor)

    def test_text_writer(self, tmp_path):
        """Test text writer function."""
        from lighter.callbacks.file_writer import write_text

        path = tmp_path / "test"
        value = "Hello, World!"

        write_text(path, value)

        assert (tmp_path / "test.txt").exists()
        content = (tmp_path / "test.txt").read_text()
        assert content == "Hello, World!"

    def test_image_2d_writer(self, tmp_path):
        """Test image 2D writer function."""
        from lighter.callbacks.file_writer import write_image_2d

        path = tmp_path / "test"
        # Create a valid 3D tensor (CHW format)
        tensor = torch.rand(3, 64, 64)

        write_image_2d(path, tensor)

        assert (tmp_path / "test.png").exists()

    def test_image_2d_writer_invalid_dimensions(self, tmp_path):
        """Test image 2D writer raises error for wrong dimensions."""
        from lighter.callbacks.file_writer import write_image_2d

        path = tmp_path / "test"
        # Create invalid 2D tensor instead of 3D
        tensor = torch.rand(64, 64)

        with pytest.raises(ValueError, match="write_image_2d expects a 3D tensor"):
            write_image_2d(path, tensor)

    def test_image_3d_writer(self, tmp_path):
        """Test image 3D writer function."""
        from lighter.callbacks.file_writer import write_image_3d

        path = tmp_path / "test"
        # Create a valid 4D tensor (CDHW format)
        tensor = torch.rand(3, 10, 64, 64)

        write_image_3d(path, tensor)

        assert (tmp_path / "test.png").exists()

    def test_image_3d_writer_invalid_dimensions(self, tmp_path):
        """Test image 3D writer raises error for wrong dimensions."""
        from lighter.callbacks.file_writer import write_image_3d

        path = tmp_path / "test"
        # Create invalid 3D tensor instead of 4D
        tensor = torch.rand(3, 64, 64)

        with pytest.raises(ValueError, match="write_image_3d expects a 4D tensor"):
            write_image_3d(path, tensor)

    def test_writer_registry_register_duplicate(self):
        """Test that registering duplicate writer raises error."""
        from lighter.callbacks.file_writer import WriterRegistry

        registry = WriterRegistry()

        @registry.register("test_writer")
        def writer1(path, value):
            pass

        with pytest.raises(ValueError, match="Writer with name 'test_writer' is already registered"):

            @registry.register("test_writer")
            def writer2(path, value):
                pass


# =============================================================================
# CsvWriter Tests
# =============================================================================


class TestCsvWriter:
    """Test suite for CsvWriter callback."""

    def test_initialization(self, tmp_path):
        """Test CsvWriter initialization."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target", "loss"])
        assert writer.path == tmp_path / "results.csv"
        assert writer.keys == ["pred", "target", "loss"]

    def test_setup(self, tmp_path, mock_trainer, mock_system):
        """Test CsvWriter setup creates temp file and writes header."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])

        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Check temp file created with header
        temp_file = tmp_path / "results.tmp_rank0.csv"
        assert temp_file.exists()
        # File is created but header not yet written (written on first write)
        # Close the file to flush the header
        writer._csv_file.close()
        content = temp_file.read_text()
        assert "pred,target" in content

    def test_write_tensor_batch(self, tmp_path, mock_trainer, mock_system):
        """Test writing a batch with tensor values."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"pred": torch.tensor([0.1, 0.2, 0.3]), "target": torch.tensor([0, 1, 0])}

        writer.write(outputs, None, 0, 0)

        # Close file to flush writes
        writer._csv_file.close()

        # Read temp file
        temp_file = tmp_path / "results.tmp_rank0.csv"
        df = pd.read_csv(temp_file)

        assert len(df) == 3
        assert list(df.columns) == ["pred", "target"]
        assert df["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
        assert df["target"].tolist() == [0, 1, 0]

    def test_write_mixed_types(self, tmp_path, mock_trainer, mock_system):
        """Test writing with mixed data types."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target", "id"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {
            "pred": torch.tensor([0.1, 0.2]),
            "target": [1, 0],  # List
            "id": ["sample1", "sample2"],  # Strings
        }

        writer.write(outputs, None, 0, 0)

        # Close file to flush writes
        writer._csv_file.close()

        temp_file = tmp_path / "results.tmp_rank0.csv"
        df = pd.read_csv(temp_file)

        assert len(df) == 2
        assert df["id"].tolist() == ["sample1", "sample2"]

    def test_write_inconsistent_lengths(self, tmp_path, mock_trainer, mock_system):
        """Test error when outputs have inconsistent lengths."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {
            "pred": torch.tensor([0.1, 0.2, 0.3]),  # 3 items
            "target": torch.tensor([0, 1]),  # 2 items
        }

        with pytest.raises(ValueError, match="inconsistent lengths"):
            writer.write(outputs, None, 0, 0)

    def test_write_missing_key(self, tmp_path, mock_trainer, mock_system):
        """Test error when required key is missing."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        outputs = {"pred": torch.tensor([0.1, 0.2])}  # Missing "target"

        with pytest.raises(KeyError, match="expected key 'target'"):
            writer.write(outputs, None, 0, 0)

    def test_get_sequence_length_tensor(self):
        """Test _get_sequence_length with tensor."""
        writer = CsvWriter(path="test.csv", keys=["pred"])

        # Scalar tensor
        assert writer._get_sequence_length(torch.tensor(5.0)) == 1

        # Vector tensor
        assert writer._get_sequence_length(torch.tensor([1, 2, 3])) == 3

    def test_get_sequence_length_list(self):
        """Test _get_sequence_length with list."""
        writer = CsvWriter(path="test.csv", keys=["pred"])
        assert writer._get_sequence_length([1, 2, 3, 4]) == 4

    def test_get_sequence_length_non_sequence(self):
        """Test _get_sequence_length with non-sequence."""
        writer = CsvWriter(path="test.csv", keys=["pred"])
        assert writer._get_sequence_length("string") is None

    def test_get_record_value_tensor(self):
        """Test _get_record_value with tensor."""
        writer = CsvWriter(path="test.csv", keys=["pred"])

        # Scalar tensor
        value = torch.tensor(5.0)
        assert writer._get_record_value(value, 0) == 5.0

        # Vector tensor
        value = torch.tensor([1.0, 2.0, 3.0])
        assert writer._get_record_value(value, 1) == 2.0

    def test_get_record_value_list(self):
        """Test _get_record_value with list."""
        writer = CsvWriter(path="test.csv", keys=["pred"])
        value = [10, 20, 30]
        assert writer._get_record_value(value, 2) == 30

    def test_on_predict_epoch_end_single_rank(self, tmp_path, mock_trainer, mock_system):
        """Test epoch end combines temp file and creates final CSV."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Write some data
        outputs = {"pred": torch.tensor([0.1, 0.2]), "target": torch.tensor([0, 1])}
        writer.write(outputs, None, 0, 0)

        # Trigger epoch end
        writer.on_predict_epoch_end(mock_trainer, mock_system)

        # Check final CSV exists
        assert (tmp_path / "results.csv").exists()
        df = pd.read_csv(tmp_path / "results.csv")
        assert len(df) == 2

        # Check temp file was removed
        temp_file = tmp_path / "results.tmp_rank0.csv"
        assert not temp_file.exists()

    def test_write_empty_outputs_raises(self, tmp_path, mock_trainer, mock_system):
        """Test write raises KeyError when outputs is empty."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Empty outputs - none of the configured keys are present
        outputs = {}

        with pytest.raises(KeyError, match="none of the configured keys"):
            writer.write(outputs, None, 0, 0)

    def test_write_no_configured_keys_present_raises(self, tmp_path, mock_trainer, mock_system):
        """Test write raises KeyError when outputs has keys but none match configured keys."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "target"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Outputs has keys, but none of them match the configured keys
        outputs = {"other_key": [1, 2, 3], "another_key": [4, 5, 6]}

        with pytest.raises(KeyError, match="none of the configured keys.*pred.*target.*were found") as exc_info:
            writer.write(outputs, None, 0, 0)

        # Verify error message includes available keys
        assert "other_key" in str(exc_info.value) or "another_key" in str(exc_info.value)

    def test_write_non_sequence_values(self, tmp_path, mock_trainer, mock_system):
        """Test write with non-sequence values (single sample)."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "label"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Single values (not sequences)
        outputs = {
            "pred": 0.95,  # Single float
            "label": "positive",  # Single string
        }

        writer.write(outputs, None, 0, 0)

        # Close file to flush
        writer._csv_file.close()

        temp_file = tmp_path / "results.tmp_rank0.csv"
        df = pd.read_csv(temp_file)

        assert len(df) == 1
        assert df["pred"].iloc[0] == 0.95
        assert df["label"].iloc[0] == "positive"

    def test_get_record_value_non_sequence(self):
        """Test _get_record_value with non-sequence value."""
        writer = CsvWriter(path="test.csv", keys=["pred"])
        value = "constant_value"
        # Should return the value as-is for any index
        assert writer._get_record_value(value, 0) == "constant_value"
        assert writer._get_record_value(value, 5) == "constant_value"

    def test_on_predict_epoch_end_before_setup(self, mock_trainer, mock_system):
        """Test on_predict_epoch_end handles case when setup wasn't called."""
        writer = CsvWriter(path="results.csv", keys=["pred"])

        # Call epoch end without setup - should return without error
        writer.on_predict_epoch_end(mock_trainer, mock_system)

    def test_close_file_closes_open_file(self, tmp_path, mock_trainer, mock_system):
        """Test _close_file closes the file handle and resets state."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # File should be open after setup
        assert writer._csv_file is not None
        assert not writer._csv_file.closed
        assert writer._csv_writer is not None

        # Call _close_file
        writer._close_file()

        # File handle should be None and writer reset
        assert writer._csv_file is None
        assert writer._csv_writer is None

    def test_close_file_handles_none_file(self, tmp_path):
        """Test _close_file handles case when file is already None."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])

        # File is None by default
        assert writer._csv_file is None

        # Should not raise
        writer._close_file()

        # Still None
        assert writer._csv_file is None

    def test_close_file_handles_already_closed_file(self, tmp_path, mock_trainer, mock_system):
        """Test _close_file handles already closed file."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Manually close the file
        writer._csv_file.close()
        assert writer._csv_file.closed

        # Should not raise when calling _close_file on closed file
        writer._close_file()

        assert writer._csv_file is None
        assert writer._csv_writer is None

    def test_on_exception_closes_file(self, tmp_path, mock_trainer, mock_system):
        """Test on_exception closes file to prevent handle leaks."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # File should be open
        assert writer._csv_file is not None
        assert not writer._csv_file.closed

        # Simulate an exception occurring
        writer.on_exception(mock_trainer, mock_system, RuntimeError("Test error"))

        # File should be closed and state reset
        assert writer._csv_file is None
        assert writer._csv_writer is None

    def test_teardown_closes_file_on_predict_stage(self, tmp_path, mock_trainer, mock_system):
        """Test teardown closes file when stage is PREDICT."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # File should be open
        assert writer._csv_file is not None

        # Call teardown with PREDICT stage
        writer.teardown(mock_trainer, mock_system, Stage.PREDICT)

        # File should be closed
        assert writer._csv_file is None
        assert writer._csv_writer is None

    def test_teardown_does_not_close_on_other_stages(self, tmp_path, mock_trainer, mock_system):
        """Test teardown does not close file for non-PREDICT stages."""
        writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred"])
        writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Store reference to check if still open
        csv_file = writer._csv_file
        assert csv_file is not None

        # Call teardown with FIT stage - should not close the file
        writer.teardown(mock_trainer, mock_system, Stage.FIT)

        # File should still be open (same reference)
        assert writer._csv_file is csv_file
        assert not writer._csv_file.closed

        # Clean up
        writer._close_file()


# =============================================================================
# Integration Tests
# =============================================================================


class TestWritersIntegration:
    """Integration tests for writers working together."""

    def test_filewriter_and_csvwriter_together(self, tmp_path, mock_trainer, mock_system):
        """Test using FileWriter and CsvWriter on same outputs."""
        file_writer = FileWriter(directory=tmp_path / "predictions", value_key="pred", writer_fn="tensor")
        csv_writer = CsvWriter(path=tmp_path / "results.csv", keys=["pred", "confidence"])

        # Setup both
        file_writer.setup(mock_trainer, mock_system, Stage.PREDICT)
        csv_writer.setup(mock_trainer, mock_system, Stage.PREDICT)

        # Write same outputs to both
        outputs = {"pred": torch.tensor([[1, 2], [3, 4]]), "confidence": torch.tensor([0.9, 0.8])}

        file_writer.write(outputs, None, 0, 0)
        csv_writer.write(outputs, None, 0, 0)

        # Check FileWriter created files
        assert (tmp_path / "predictions" / "0.pt").exists()
        assert (tmp_path / "predictions" / "1.pt").exists()

        # Check CsvWriter created temp file
        temp_csv = tmp_path / "results.tmp_rank0.csv"
        assert temp_csv.exists()
