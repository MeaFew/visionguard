#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace visionguard {

// Input shape expected by the YOLOv8 ONNX model.
struct InputShape {
    std::int64_t batch = 1;
    std::int64_t channels = 3;
    std::int64_t height = 640;
    std::int64_t width = 640;
};

// Letterbox padding information to reverse on output coordinates.
struct LetterboxInfo {
    float scale = 1.0f;
    int pad_top = 0;
    int pad_left = 0;
    int original_width = 0;
    int original_height = 0;
};

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(const InputShape& input_shape);

    // Preprocess an image for YOLOv8 ONNX inference:
    // 1. Resize with letterbox padding
    // 2. Normalize to [0, 1]
    // 3. Convert BGR -> RGB
    // 4. Layout: HWC -> CHW
    // Returns flattened float tensor data and letterbox info.
    std::vector<float> preprocess(const cv::Mat& image, LetterboxInfo& info) const;

    // Convert a single CHW image back to a cv::Mat (for debugging).
    cv::Mat chw_to_mat(const std::vector<float>& chw_data) const;

private:
    InputShape input_shape_;
};

}  // namespace visionguard
