#include "image_preprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

namespace visionguard {

ImagePreprocessor::ImagePreprocessor(const InputShape& input_shape)
    : input_shape_(input_shape) {}

std::vector<float> ImagePreprocessor::preprocess(
    const cv::Mat& image,
    LetterboxInfo& info) const {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    info.original_width = image.cols;
    info.original_height = image.rows;

    const int target_h = static_cast<int>(input_shape_.height);
    const int target_w = static_cast<int>(input_shape_.width);

    // Compute scale preserving aspect ratio.
    const float scale_x = static_cast<float>(target_w) / static_cast<float>(image.cols);
    const float scale_y = static_cast<float>(target_h) / static_cast<float>(image.rows);
    info.scale = std::min(scale_x, scale_y);

    const int new_w = static_cast<int>(image.cols * info.scale);
    const int new_h = static_cast<int>(image.rows * info.scale);

    if (new_w <= 0 || new_h <= 0) {
        throw std::invalid_argument("Computed resize dimensions must be positive");
    }

    info.pad_left = (target_w - new_w) / 2;
    info.pad_top = (target_h - new_h) / 2;

    // Resize image.
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Create padded canvas.
    cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat roi = padded(cv::Rect(info.pad_left, info.pad_top, new_w, new_h));
    resized.copyTo(roi);

    // Convert BGR -> RGB, normalize to [0, 1], and flatten as CHW.
    std::vector<float> tensor(input_shape_.channels * target_h * target_w);
    constexpr float norm = 1.0f / 255.0f;

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            const cv::Vec3b& pixel = padded.at<cv::Vec3b>(y, x);
            // R, G, B
            tensor[0 * target_h * target_w + y * target_w + x] = pixel[2] * norm;
            tensor[1 * target_h * target_w + y * target_w + x] = pixel[1] * norm;
            tensor[2 * target_h * target_w + y * target_w + x] = pixel[0] * norm;
        }
    }

    return tensor;
}

cv::Mat ImagePreprocessor::chw_to_mat(const std::vector<float>& chw_data) const {
    const int h = static_cast<int>(input_shape_.height);
    const int w = static_cast<int>(input_shape_.width);
    cv::Mat mat(h, w, CV_8UC3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int r = static_cast<int>(chw_data[0 * h * w + y * w + x] * 255.0f);
            const int g = static_cast<int>(chw_data[1 * h * w + y * w + x] * 255.0f);
            const int b = static_cast<int>(chw_data[2 * h * w + y * w + x] * 255.0f);
            mat.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(std::clamp(b, 0, 255)),
                static_cast<uchar>(std::clamp(g, 0, 255)),
                static_cast<uchar>(std::clamp(r, 0, 255)));
        }
    }

    return mat;
}

}  // namespace visionguard
