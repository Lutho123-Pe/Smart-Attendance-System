import numpy as np
import pytest

from recognition import FaceEngine, ModelUnavailable


def test_model_files_are_required_for_recognition(tmp_path):
    with pytest.raises(ModelUnavailable, match="Install the YuNet and SFace"):
        FaceEngine(tmp_path / "missing-yunet.onnx", tmp_path / "missing-sface.onnx")


def test_template_average_is_normalized():
    vectors = [np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)]
    average = FaceEngine.average(vectors)
    assert np.isclose(np.linalg.norm(average), 1.0)


def test_template_average_rejects_empty_input():
    with pytest.raises(ValueError):
        FaceEngine.average([])