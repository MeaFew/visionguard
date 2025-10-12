#pragma once

#include "image_preprocessor.h"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace visionguard {

// A detection result from the inference engine.
struct Detection {
    int class_id = 0;
    std::string class_name;
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

class InferenceEngine {
public:
    InferenceEngine(
        const std::string& model_path,
        const std::vector<std::string>& class_names,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f,
        int num_threads = 4,
        const InputShape& input_shape = InputShape{});

    // Run inference on a single image.
    std::vector<Detection> detect(const cv::Mat& image);

    // Run inference on already preprocessed data (for benchmarking).
    std::vector<Detection> infer_from_tensor(
        const std::vector<float>& input_tensor,
        const LetterboxInfo& info,
        int original_width,
        int original_height);

    const std::string& model_path() const { return model_path_; }

private:
    // Decode YOLOv8 ONNX output (1 x (4 + C) x 8400) into detection boxes.
    std::vector<Detection> decode_output(
        const std::vector<float>& output_data,
        const std::vector<std::int64_t>& output_shape,
        const LetterboxInfo& info,
        int original_width,
        int original_height);

    std::string model_path_;
    std::vector<std::string> class_names_;
    float conf_threshold_;
    float iou_threshold_;
    InputShape input_shape_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<std::int64_t> input_shape_vec_;
};

}  // namespace visionguard
