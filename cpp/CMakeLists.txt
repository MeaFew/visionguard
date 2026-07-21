cmake_minimum_required(VERSION 3.20)
project(visionguard_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build C++ unit tests" ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(ONNXRuntime QUIET)
if(NOT ONNXRuntime_FOUND)
    find_package(onnxruntime QUIET)
endif()
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

if(NOT TARGET ONNXRuntime::ONNXRuntime AND NOT TARGET onnxruntime)
    message(FATAL_ERROR "ONNX Runtime target not found. Install ONNX Runtime and set CMAKE_PREFIX_PATH.")
endif()

# Protobuf / gRPC generation
set(PROTO_FILE proto/visionguard.proto)
get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)

set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc")
set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h")
set(GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.cc")
set(GRPC_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
         --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
         -I ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
)

add_library(visionguard_proto
    ${PROTO_SRC}
    ${GRPC_SRC}
)
target_link_libraries(visionguard_proto
    PUBLIC gRPC::grpc++
    PUBLIC protobuf::libprotobuf
)
target_include_directories(visionguard_proto
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

# Core inference library
add_library(visionguard_inference
    src/image_preprocessor.cpp
    src/inference_engine.cpp
)
target_include_directories(visionguard_inference
    PUBLIC include
)
if(TARGET ONNXRuntime::ONNXRuntime)
    set(ONNX_RUNTIME_TARGET ONNXRuntime::ONNXRuntime)
else()
    set(ONNX_RUNTIME_TARGET onnxruntime)
endif()

target_link_libraries(visionguard_inference
    PUBLIC ${OpenCV_LIBS}
    PUBLIC ${ONNX_RUNTIME_TARGET}
)

# gRPC server library
add_library(visionguard_server_lib
    src/grpc_server.cpp
)
target_include_directories(visionguard_server_lib
    PUBLIC include
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(visionguard_server_lib
    PUBLIC visionguard_inference
    PUBLIC visionguard_proto
)

# Server executable
add_executable(visionguard_server src/main.cpp)
target_link_libraries(visionguard_server
    PRIVATE visionguard_server_lib
)

# Tests
# TODO: wire these Catch2 tests into CI (.github/workflows/ci.yml) once a
# runner image with the C++ dependencies (OpenCV, ONNX Runtime, gRPC) is
# available; currently they only run locally via BUILD_TESTS=ON.
if(BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(visionguard_tests tests/test_inference.cpp)
        target_link_libraries(visionguard_tests
            PRIVATE visionguard_inference
            PRIVATE Catch2::Catch2WithMain
        )
    else()
        message(STATUS "Catch2 not found; skipping C++ tests")
    endif()
endif()

# Install
install(TARGETS visionguard_server DESTINATION bin)
