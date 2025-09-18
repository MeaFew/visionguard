#include <image_preprocessor.h>

#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

using namespace visionguard;

TEST_CASE("ImagePreprocessor preserves input shape") {
    ImagePreprocessor preprocessor(InputShape{1, 3, 640, 640});
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));

    LetterboxInfo info{};
    std::vector<float> tensor = preprocessor.preprocess(image, info);

    REQUIRE(tensor.size() == 1 * 3 * 640 * 640);
    REQUIRE(info.original_width == 640);
    REQUIRE(info.original_height == 480);
    REQUIRE(info.scale > 0.0f);
}

TEST_CASE("ImagePreprocessor rejects empty image") {
    ImagePreprocessor preprocessor(InputShape{1, 3, 640, 640});
    cv::Mat empty;
    LetterboxInfo info{};

    REQUIRE_THROWS_AS(preprocessor.preprocess(empty, info), std::invalid_argument);
}

TEST_CASE("Letterbox padding centers the image") {
    ImagePreprocessor preprocessor(InputShape{1, 3, 640, 640});
    cv::Mat image(320, 640, CV_8UC3, cv::Scalar(0, 0, 255));

    LetterboxInfo info{};
    preprocessor.preprocess(image, info);

    REQUIRE(info.pad_top > 0);
    REQUIRE(info.pad_left == 0);
}
