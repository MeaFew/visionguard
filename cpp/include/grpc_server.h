#pragma once

#include "inference_engine.h"

#include <grpcpp/grpcpp.h>

#include "visionguard.grpc.pb.h"
#include "visionguard.pb.h"

#include <memory>
#include <string>

namespace visionguard {

class VisionGuardServiceImpl final : public VisionGuard::Service {
public:
    explicit VisionGuardServiceImpl(std::shared_ptr<InferenceEngine> engine);

    grpc::Status Detect(
        grpc::ServerContext* context,
        const DetectRequest* request,
        DetectResponse* response) override;

    grpc::Status Health(
        grpc::ServerContext* context,
        const HealthRequest* request,
        HealthResponse* response) override;

    grpc::Status Benchmark(
        grpc::ServerContext* context,
        const BenchmarkRequest* request,
        BenchmarkResponse* response) override;

private:
    std::shared_ptr<InferenceEngine> engine_;
};

// Run the gRPC server on the given address.
void run_grpc_server(
    const std::string& address,
    std::shared_ptr<InferenceEngine> engine);

}  // namespace visionguard
