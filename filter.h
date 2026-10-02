#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst);   // custom greyscale: each pixel = brightest of its B, G, R values
int sepia(cv::Mat &src, cv::Mat &dst, bool vignette = false);   // antique sepia tone, optional dark edges
int blur5x5_1(cv::Mat &src, cv::Mat &dst);   // naive 5x5 Gaussian blur using .at<>
int blur5x5_2(cv::Mat &src, cv::Mat &dst);   // faster separable 5x5 Gaussian blur using row pointers
int sobelX3x3(cv::Mat &src, cv::Mat &dst);   // 3x3 Sobel X, positive right, output CV_16SC3
int sobelY3x3(cv::Mat &src, cv::Mat &dst);   // 3x3 Sobel Y, positive up, output CV_16SC3
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);   // gradient magnitude sqrt(sx^2 + sy^2), output CV_8UC3
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 10);   // blur then quantize each channel into 'levels' values
int depthFocus(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int lo = 110, int hi = 170);   // sharp near, blurred grey far
int colourPop(cv::Mat &src, cv::Mat &dst);   // pixel-wise: strong reds keep colour, the rest turns grey
int cartoon(cv::Mat &src, cv::Mat &dst, int levels = 10);   // Winnemoeller-style abstraction: bilateral + DoG edges + soft quantization
int faceSpotlight(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst, int minWidth = 50);   // faces in colour, rest dim grey
int canny(cv::Mat &src, cv::Mat &dst, int low = 20, int high = 50);   // from-scratch Canny: thin white edges on black

#endif
