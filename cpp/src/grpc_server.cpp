#include "grpc_server.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace visionguard {

namespace {

std::vector<std::string> default_class_names() {
    return {
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    };
}

}  // namespace

VisionGuardServiceImpl::VisionGuardServiceImpl(std::shared_ptr<InferenceEngine> engine)
    : engine_(std::move(engine)) {}

grpc::Status VisionGuardServiceImpl::Detect(
    grpc::ServerContext* /*context*/,
    const DetectRequest* request,
    DetectResponse* response) {
    if (request->image_data().empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "image_data is empty");
    }

    std::vector<uchar> buffer(request->image_data().begin(), request->image_data().end());
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "failed to decode image");
    }

    try {
        std::vector<Detection> dets = engine_->detect(image);
        for (const auto& d : dets) {
            Detection* out = response->add_detections();
            out->set_class_id(d.class_id);
            out->set_class_name(d.class_name);
            out->set_confidence(d.confidence);
            out->set_x1(d.x1);
            out->set_y1(d.y1);
            out->set_x2(d.x2);
            out->set_y2(d.y2);
        }
        response->set_image_width(image.cols);
        response->set_image_height(image.rows);
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }

    return grpc::Status::OK;
}

grpc::Status VisionGuardServiceImpl::Health(
    grpc::ServerContext* /*context*/,
    const HealthRequest* /*request*/,
    HealthResponse* response) {
    response->set_status("ok");
    response->set_model_path(engine_->model_path());
    return grpc::Status::OK;
}

grpc::Status VisionGuardServiceImpl::Benchmark(
    grpc::ServerContext* /*context*/,
    const BenchmarkRequest* request,
    BenchmarkResponse* response) {
    const int iterations = request->iterations() > 0 ? request->iterations() : 100;
    constexpr int dummy_h = 640;
    constexpr int dummy_w = 640;

    cv::Mat dummy(dummy_h, dummy_w, CV_8UC3, cv::Scalar(114, 114, 114));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        try {
            engine_->detect(dummy);
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    const float total_ms = std::chrono::duration<float, std::milli>(end - start).count();
    response->set_iterations(iterations);
    response->set_total_ms(total_ms);
    response->set_avg_ms(total_ms / static_cast<float>(iterations));

    return grpc::Status::OK;
}

void run_grpc_server(
    const std::string& address,
    std::shared_ptr<InferenceEngine> engine) {
    VisionGuardServiceImpl service(std::move(engine));

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("Failed to start gRPC server on " + address);
    }

    std::cout << "VisionGuard gRPC server listening on " << address << std::endl;
    server->Wait();
}

}  // namespace visionguard
