"""Traditional image preprocessing pipeline using OpenCV.

.. note::
    Color inputs are assumed to be in BGR order (OpenCV default) unless
    otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass
class Blob:
    """Represents a detected blob/connected component."""

    area: float
    perimeter: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


class Preprocessor:
    """OpenCV-based image preprocessing for defect images.

    Color inputs are assumed to be in BGR order (OpenCV default) unless
    otherwise noted.
    """

    def _validate_kernel(
        self, kernel_size: tuple[int, int] | int, name: str = "kernel_size"
    ) -> None:
        """Validate that a kernel size has positive odd integer values."""
        if isinstance(kernel_size, int):
            sizes = (kernel_size,)
        elif isinstance(kernel_size, tuple):
            sizes = kernel_size
        else:
            raise ValueError(
                f"{name} must be an int or a tuple of two ints, got {type(kernel_size)}"
            )

        for value in sizes:
            if not isinstance(value, int) or value <= 0 or value % 2 == 0:
                raise ValueError(
                    f"{name} values must be positive odd integers, got {kernel_size}"
                )

    def gaussian_blur(
        self, image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma: float = 1.0
    ) -> np.ndarray:
        """Apply Gaussian blur for noise reduction.

        Color inputs are assumed to be in BGR order (OpenCV default).
        """
        self._validate_kernel(kernel_size)
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median blur to remove salt-and-pepper noise.

        Color inputs are assumed to be in BGR order (OpenCV default).
        """
        self._validate_kernel(kernel_size)
        return cv2.medianBlur(image, kernel_size)

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE on luminance channel."""
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def edge_detection(
        self,
        image: np.ndarray,
        method: Literal["canny", "sobel"] = "canny",
        threshold1: float = 50,
        threshold2: float = 150,
    ) -> np.ndarray:
        """Detect edges using Canny or Sobel.

        Color inputs are assumed to be in BGR order (OpenCV default).
        """
        if method == "canny":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return cv2.Canny(gray, threshold1, threshold2)

        if method == "sobel":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(np.sqrt(sobelx**2 + sobely**2))

        raise ValueError(f"Unknown edge detection method: {method}")

    def morphology(
        self,
        image: np.ndarray,
        operation: Literal["open", "close", "erode", "dilate"] = "open",
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """Apply morphological operations.

        Color inputs are assumed to be in BGR order (OpenCV default).
        """
        self._validate_kernel(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        ops = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
        }
        if operation not in ops:
            raise ValueError(f"Unknown morphology operation: {operation}")
        return cv2.morphologyEx(image, ops[operation], kernel, iterations=iterations)

    def blob_analysis(self, binary_image: np.ndarray) -> list[Blob]:
        """Perform connected component analysis and return blob descriptors.

        Args:
            binary_image: A 2-D uint8 numpy array containing the binary image.
        """
        if (
            not isinstance(binary_image, np.ndarray)
            or binary_image.ndim != 2
            or binary_image.dtype != np.uint8
        ):
            raise ValueError("binary_image must be a 2-D uint8 numpy array")

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        blobs: list[Blob] = []
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            # Crop to the component's bounding box for faster contour extraction.
            roi = labels[y : y + h, x : x + w]
            mask = (roi == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
            blobs.append(
                Blob(
                    area=float(area),
                    perimeter=float(perimeter),
                    centroid=(float(cx), float(cy)),
                    bbox=(int(x), int(y), int(w), int(h)),
                )
            )

        return blobs

    def frequency_filter(
        self,
        image: np.ndarray,
        filter_type: Literal["lowpass", "highpass"] = "lowpass",
        cutoff: int = 30,
    ) -> np.ndarray:
        """Apply simple FFT-based frequency domain filter.

        Color inputs are assumed to be in BGR order (OpenCV default).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        rows, cols = gray.shape
        max_cutoff = min(rows, cols) // 2
        if cutoff <= 0 or cutoff > max_cutoff:
            raise ValueError(
                f"cutoff must be in the range (0, {max_cutoff}] for this image, got {cutoff}"
            )

        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)

        if filter_type == "lowpass":
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 1
        elif filter_type == "highpass":
            mask[:] = 1
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 0
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        filtered = dft_shift * mask
        filtered_shift = np.fft.ifftshift(filtered)
        restored = np.fft.ifft2(filtered_shift)
        normalized = cv2.normalize(np.abs(restored), None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
