"""Tests for OpenCV preprocessor."""

import numpy as np
import pytest

from visionguard.core.preprocessor import Preprocessor, letterbox_tensor


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray() -> np.ndarray:
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


def test_gaussian_blur(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.gaussian_blur(sample_image)
    assert blurred.shape == sample_image.shape


def test_gaussian_blur_grayscale(sample_gray: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.gaussian_blur(sample_gray)
    assert blurred.shape == sample_gray.shape


def test_gaussian_blur_invalid_kernel(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.gaussian_blur(sample_image, kernel_size=(4, 4))


def test_median_blur(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.median_blur(sample_image)
    assert blurred.shape == sample_image.shape


def test_median_blur_grayscale(sample_gray: np.ndarray) -> None:
    proc = Preprocessor()
    blurred = proc.median_blur(sample_gray)
    assert blurred.shape == sample_gray.shape


def test_median_blur_invalid_kernel(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.median_blur(sample_image, kernel_size=4)


def test_histogram_equalization(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    enhanced = proc.histogram_equalization(sample_image)
    assert enhanced.shape == sample_image.shape


def test_edge_detection_canny(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    edges = proc.edge_detection(sample_image, method="canny")
    assert len(edges.shape) == 2


def test_edge_detection_sobel(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    edges = proc.edge_detection(sample_image, method="sobel")
    assert len(edges.shape) == 2


def test_edge_detection_grayscale(sample_gray: np.ndarray) -> None:
    proc = Preprocessor()
    edges = proc.edge_detection(sample_gray, method="canny")
    assert edges.shape == sample_gray.shape


def test_edge_detection_invalid_method(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.edge_detection(sample_image, method="laplacian")


def test_morphology(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    opened = proc.morphology(sample_image, operation="open")
    assert opened.shape == sample_image.shape


def test_morphology_invalid_operation(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.morphology(sample_image, operation="invalid")


@pytest.mark.parametrize("kernel_size", [2, 0, -1])
def test_morphology_invalid_kernel(sample_image: np.ndarray, kernel_size: int) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.morphology(sample_image, kernel_size=kernel_size)


def test_blob_analysis() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(binary, (10, 10), (20, 20), 255, -1)

    proc = Preprocessor()
    blobs = proc.blob_analysis(binary)
    assert len(blobs) == 1
    assert blobs[0].area > 0


def test_blob_analysis_empty() -> None:
    binary = np.zeros((50, 50), dtype=np.uint8)
    proc = Preprocessor()
    blobs = proc.blob_analysis(binary)
    assert blobs == []


def test_blob_analysis_invalid_input() -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.blob_analysis(np.zeros((50, 50), dtype=np.float32))
    with pytest.raises(ValueError):
        proc.blob_analysis(np.zeros((50, 50, 3), dtype=np.uint8))


def test_frequency_filter(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    filtered = proc.frequency_filter(sample_image, filter_type="lowpass", cutoff=10)
    assert filtered.shape[:2] == sample_image.shape[:2]


def test_frequency_filter_grayscale(sample_gray: np.ndarray) -> None:
    proc = Preprocessor()
    filtered = proc.frequency_filter(sample_gray, filter_type="highpass", cutoff=5)
    assert filtered.shape == sample_gray.shape


def test_frequency_filter_invalid_filter_type(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.frequency_filter(sample_image, filter_type="bandpass", cutoff=10)


def test_frequency_filter_invalid_cutoff(sample_image: np.ndarray) -> None:
    proc = Preprocessor()
    with pytest.raises(ValueError):
        proc.frequency_filter(sample_image, cutoff=0)
    with pytest.raises(ValueError):
        proc.frequency_filter(sample_image, cutoff=200)


def test_letterbox_tensor() -> None:
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor, scale, pad_left, pad_top = letterbox_tensor(image, size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0 < scale <= 1.0
    assert pad_left >= 0
    assert pad_top >= 0


def test_letterbox_tensor_invalid_image() -> None:
    with pytest.raises(ValueError):
        letterbox_tensor(np.zeros((100, 100), dtype=np.uint8), size=640)
