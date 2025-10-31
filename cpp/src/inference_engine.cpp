#include "inference_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace visionguard {

namespace {

// Build configured Ort::SessionOptions before constructing the Ort::Session.
Ort::SessionOptions make_session_options(int num_threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

// Greedy NMS on [x1, y1, x2, y2] boxes.
std::vector<int> nms(
    const std::vector<Detection>& detections,
    float iou_threshold) {
    const int n = static_cast<int>(detections.size());
    if (n == 0) return {};

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i) {
        const auto& d = detections[i];
        areas[i] = (d.x2 - d.x1) * (d.y2 - d.y1);
    }

    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);

        const auto& d_i = detections[idx];
        for (int j = 0; j < n; ++j) {
            if (suppressed[j] || j == idx) continue;
            const auto& d_j = detections[j];

            const float x1 = std::max(d_i.x1, d_j.x1);
            const float y1 = std::max(d_i.y1, d_j.y1);
            const float x2 = std::min(d_i.x2, d_j.x2);
            const float y2 = std::min(d_i.y2, d_j.y2);

            const float inter_w = std::max(0.0f, x2 - x1);
            const float inter_h = std::max(0.0f, y2 - y1);
            const float inter = inter_w * inter_h;
            const float union_area = areas[idx] + areas[j] - inter + 1e-6f;
            const float iou = inter / union_area;

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

}  // namespace

InferenceEngine::InferenceEngine(
    const std::string& model_path,
    const std::vector<std::string>& class_names,
    float conf_threshold,
    float iou_threshold,
    int num_threads,
    const InputShape& input_shape)
    : model_path_(model_path),
      class_names_(class_names),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      input_shape_(input_shape),
      env_(ORT_LOGGING_LEVEL_WARNING, "VisionGuard"),
      session_options_(make_session_options(num_threads)),
      session_(env_, model_path_.c_str(), session_options_),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    // Get input/output node names (store as std::string to keep memory alive).
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t num_inputs = session_.GetInputCount();
    for (std::size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_.GetInputNameAllocated(i, allocator);
        input_node_names_.emplace_back(name_ptr.get());
    }

    const std::size_t num_outputs = session_.GetOutputCount();
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_.GetOutputNameAllocated(i, allocator);
        output_node_names_.emplace_back(name_ptr.get());
    }

    input_shape_vec_ = {
        input_shape_.batch,
        input_shape_.channels,
        input_shape_.height,
        input_shape_.width,
    };
}

std::vector<Detection> InferenceEngine::detect(const cv::Mat& image) {
    ImagePreprocessor preprocessor(input_shape_);
    LetterboxInfo info{};
    std::vector<float> input_tensor = preprocessor.preprocess(image, info);
    return infer_from_tensor(input_tensor, info, image.cols, image.rows);
}

std::vector<Detection> InferenceEngine::infer_from_tensor(
    const std::vector<float>& input_tensor,
    const LetterboxInfo& info,
    int original_width,
    int original_height) {
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_tensor.data()),
        input_tensor.size(),
        input_shape_vec_.data(),
        input_shape_vec_.size());

    std::vector<const char*> input_names_cstr;
    input_names_cstr.reserve(input_node_names_.size());
    for (const auto& name : input_node_names_) {
        input_names_cstr.push_back(name.c_str());
    }

    std::vector<const char*> output_names_cstr;
    output_names_cstr.reserve(output_node_names_.size());
    for (const auto& name : output_node_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(),
        &input_tensor_ort,
        input_names_cstr.size(),
        output_names_cstr.data(),
        output_names_cstr.size());

    Ort::Value& output_tensor = output_tensors.front();
    const auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    std::int64_t output_count = 1;
    for (auto dim : output_shape) {
        output_count *= dim;
    }

    float* raw_output = output_tensor.GetTensorMutableData<float>();
    std::vector<float> output_data(raw_output, raw_output + output_count);

    return decode_output(output_data, output_shape, info, original_width, original_height);
}

std::vector<Detection> InferenceEngine::decode_output(
    const std::vector<float>& output_data,
    const std::vector<std::int64_t>& output_shape,
    const LetterboxInfo& info,
    int original_width,
    int original_height) {
    if (output_shape.size() != 3) {
        throw std::runtime_error("Unexpected output tensor rank");
    }
    const int num_classes = static_cast<int>(class_names_.size());
    const int rows = static_cast<int>(output_shape[2]);  // 8400
    const int cols = static_cast<int>(output_shape[1]);  // 84

    std::vector<Detection> candidates;
    candidates.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        const float* row = output_data.data() + i * cols;
        const float cx = row[0];
        const float cy = row[1];
        const float w = row[2];
        const float h = row[3];

        // Find best class.
        int best_class = 0;
        float best_score = row[4];
        for (int c = 1; c < num_classes; ++c) {
            const float score = row[4 + c];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold_) {
            continue;
        }

        // YOLOv8 ONNX output uses pixel coordinates in the 640x640 letterbox frame.
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Remove letterbox padding.
        x1 -= static_cast<float>(info.pad_left);
        y1 -= static_cast<float>(info.pad_top);
        x2 -= static_cast<float>(info.pad_left);
        y2 -= static_cast<float>(info.pad_top);

        // Rescale to original image size.
        x1 /= info.scale;
        y1 /= info.scale;
        x2 /= info.scale;
        y2 /= info.scale;

        // Clamp to image bounds.
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height)));

        Detection det;
        det.class_id = best_class;
        det.class_name = best_class < static_cast<int>(class_names_.size()) ? class_names_[best_class] : "unknown";
        det.confidence = best_score;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        candidates.push_back(det);
    }

    // Apply NMS.
    std::vector<int> keep = nms(candidates, iou_threshold_);
    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        result.push_back(candidates[idx]);
    }

    return result;
}

}  // namespace visionguard
