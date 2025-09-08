#include "inference_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace visionguard {

namespace {

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
      session_options_(),
      session_(env_, model_path_.c_str(), session_options_),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Get input/output node names.
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t num_inputs = session_.GetInputCount();
    for (std::size_t i = 0; i < num_inputs; ++i) {
        input_node_names_.push_back(session_.GetInputNameAllocated(i, allocator).get());
    }

    const std::size_t num_outputs = session_.GetOutputCount();
    for (std::size_t i = 0; i < num_outputs; ++i) {
        output_node_names_.push_back(session_.GetOutputNameAllocated(i, allocator).get());
    }

    input_shape_vec_ = {
        input_shape_.batch,
        input_shape_.channels,
        input_shape_.height,
        input_shape_.width,
    };

    // Typical YOLOv8 ONNX output shape: 1 x 84 x 8400.
    output_shape_vec_ = {
        input_shape_.batch,
        4 + static_cast<std::int64_t>(class_names_.size()),
        8400,
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

    Ort::Value output_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info_,
        output_shape_vec_.data(),
        output_shape_vec_.size());

    session_.Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor_ort,
        input_node_names_.size(),
        output_node_names_.data(),
        &output_tensor_ort,
        output_node_names_.size());

    std::vector<float> output_data(output_tensor_ort.GetTensorMutableData<float>(),
                                   output_tensor_ort.GetTensorMutableData<float>() +
                                       output_shape_vec_[0] * output_shape_vec_[1] * output_shape_vec_[2]);

    return decode_output(output_data, info, original_width, original_height);
}

std::vector<Detection> InferenceEngine::decode_output(
    const std::vector<float>& output_data,
    const LetterboxInfo& info,
    int original_width,
    int original_height) {
    const int num_classes = static_cast<int>(class_names_.size());
    const int rows = static_cast<int>(output_shape_vec_[2]);  // 8400
    const int cols = static_cast<int>(output_shape_vec_[1]);  // 84

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

        // Convert from normalized xywh to original image xyxy.
        const float model_w = static_cast<float>(input_shape_.width);
        const float model_h = static_cast<float>(input_shape_.height);

        float x1 = (cx - w / 2.0f) * model_w;
        float y1 = (cy - h / 2.0f) * model_h;
        float x2 = (cx + w / 2.0f) * model_w;
        float y2 = (cy + h / 2.0f) * model_h;

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
