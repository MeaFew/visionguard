#include "grpc_server.h"
#include "inference_engine.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> neu_det_classes() {
    return {
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    };
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "Options:\n"
              << "  --model <path>      Path to ONNX model (default: runs/detect/models/real_train/weights/best.onnx)\n"
              << "  --address <addr>    gRPC listen address (default: 0.0.0.0:50051)\n"
              << "  --conf <threshold>  Confidence threshold (default: 0.25)\n"
              << "  --iou <threshold>   IoU threshold for NMS (default: 0.45)\n"
              << "  --threads <n>       ONNX intra-op threads (default: 4)\n"
              << "  --help              Show this message\n";
}

void print_arg_error(
    const char* program,
    const char* arg,
    const char* value,
    const char* expected) {
    std::cerr << "Invalid value for " << arg << ": '" << value
              << "' (expected " << expected << ")" << std::endl;
    print_usage(program);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "runs/detect/models/real_train/weights/best.onnx";
    std::string address = "0.0.0.0:50051";
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int num_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            try {
                conf_threshold = std::stof(argv[++i]);
            } catch (const std::exception&) {
                print_arg_error(argv[0], "--conf", argv[i], "a float");
                return 1;
            }
        } else if (arg == "--iou" && i + 1 < argc) {
            try {
                iou_threshold = std::stof(argv[++i]);
            } catch (const std::exception&) {
                print_arg_error(argv[0], "--iou", argv[i], "a float");
                return 1;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            try {
                num_threads = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                print_arg_error(argv[0], "--threads", argv[i], "an integer");
                return 1;
            }
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        auto engine = std::make_shared<visionguard::InferenceEngine>(
            model_path,
            neu_det_classes(),
            conf_threshold,
            iou_threshold,
            num_threads);

        std::cout << "Loaded model: " << model_path << std::endl;
        visionguard::run_grpc_server(address, std::move(engine));
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
