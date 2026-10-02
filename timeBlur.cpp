#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "filter.h"            // blur5x5_1 and blur5x5_2

// Times both blur functions on an image.  Usage: ./timeBlur [image]  (default data/image1.jpg)
int main(int argc, char *argv[]) {
    const char *path = argc > 1 ? argv[1] : "data/image1.jpg";
    cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);   // load the test image
    if (src.empty()) {
        std::cout << "Could not read " << path << std::endl;
        return -1;
    }
    cv::Mat dst;
    const int N = 10;          // run each version N times and report the average

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) blur5x5_1(src, dst);
    auto t1 = std::chrono::steady_clock::now();
    cv::imwrite("data/blur1.jpg", dst);                  // save so you can check it looks right

    for (int i = 0; i < N; i++) blur5x5_2(src, dst);
    auto t2 = std::chrono::steady_clock::now();
    cv::imwrite("data/blur2.jpg", dst);

    double ms1 = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    double ms2 = std::chrono::duration<double, std::milli>(t2 - t1).count() / N;
    std::cout << "Image: " << path << " (" << src.cols << "x" << src.rows << ")" << std::endl;
    std::cout << "blur5x5_1: " << ms1 << " ms per image" << std::endl;
    std::cout << "blur5x5_2: " << ms2 << " ms per image" << std::endl;
    std::cout << "Speed-up: " << ms1 / ms2 << "x" << std::endl;
    return 0;
}
