#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "filter.h"            // our own image filters (filter.cpp), reused as-is
#include "faceDetect.h"        // provided face detection code (faceDetect.cpp)
#include "da2-code/DA2Network.hpp"  // provided Depth Anything V2 wrapper (needs ONNX Runtime)

// Filter stacking: each key ADDS a filter to a chain, and every frame the filters are
// applied one after another, so each filter works on the output of the previous one.
// The chain is written at the bottom of the video, e.g. "Sepia + Blur + Canny".
//   same keys as vidDisplay add a filter, 'u' = undo last filter, 'c' = clear chain,
//   'f' = face boxes on/off, 's' = save picture, 'q' = quit

// readable name for each filter key (empty = not a filter key)
static std::string filterName(char key) {
    switch (key) {
        case 'g': return "OpenCV Grey";
        case 'h': return "Custom Grey";
        case 'p': return "Sepia";
        case 'v': return "Sepia Vignette";
        case 'b': return "Blur";
        case 'x': return "Sobel X";
        case 'y': return "Sobel Y";
        case 'm': return "Magnitude";
        case 'l': return "Blur Quantize";
        case 'd': return "Depth Map";
        case 'z': return "Depth Focus";
        case 'r': return "Colour Pop";
        case 't': return "Cartoon";
        case 'o': return "Face Spotlight";
        case 'e': return "Canny";
        default:  return "";
    }
}

// apply ONE filter to img (in place). Every filter leaves img as 8-bit 3-channel,
// so the next filter in the chain can always accept it.
//   frame/depth/faces come from the ORIGINAL camera frame, because the depth network
//   and the face detector work badly on an already-filtered picture.
static void applyFilter(char key, cv::Mat &img, cv::Mat &depth, std::vector<cv::Rect> &faces) {
    cv::Mat out, sx, sy;       // out = result of this step (filters don't like src == dst)
    if (key == 'g') {
        cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);   // 1 channel...
        cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);   // ...back to 3 so the next filter accepts it
    }
    else if (key == 'h') greyscale(img, out);
    else if (key == 'p') sepia(img, out);
    else if (key == 'v') sepia(img, out, true);
    else if (key == 'b') blur5x5_2(img, out);
    else if (key == 'x') { sobelX3x3(img, sx); cv::convertScaleAbs(sx, out); }   // signed 16-bit -> 8-bit
    else if (key == 'y') { sobelY3x3(img, sy); cv::convertScaleAbs(sy, out); }
    else if (key == 'm') { sobelX3x3(img, sx); sobelY3x3(img, sy); magnitude(sx, sy, out); }
    else if (key == 'l') blurQuantize(img, out, 10);
    else if (key == 'd') cv::applyColorMap(depth, out, cv::COLORMAP_INFERNO);   // depth map replaces the picture
    else if (key == 'z') depthFocus(img, depth, out);                          // blur the far part of the CURRENT picture
    else if (key == 'r') colourPop(img, out);
    else if (key == 't') cartoon(img, out, 10);
    else if (key == 'o') faceSpotlight(img, faces, out);
    else if (key == 'e') canny(img, out, 5, 50);
    else return;               // unknown key: leave img unchanged
    img = out;
}

// write the chain ("Sepia + Blur + Canny") centred at the bottom on a dark strip
static void drawChain(cv::Mat &img, const std::vector<char> &chain) {
    std::string text = "Original";
    if (!chain.empty()) {
        text = filterName(chain[0]);
        for (size_t i = 1; i < chain.size(); i++) text += " + " + filterName(chain[i]);
    }

    int font = cv::FONT_HERSHEY_SIMPLEX, thick = 2, base = 0;
    double scale = img.rows / 700.0;                              // bigger text on bigger frames
    cv::Size sz = cv::getTextSize(text, font, scale, thick, &base);
    if (sz.width > img.cols - 20) {                                // long chain: shrink so it still fits
        scale *= (img.cols - 20.0) / sz.width;
        sz = cv::getTextSize(text, font, scale, thick, &base);
    }

    int stripH = sz.height + base + 20;                            // height of the dark strip
    cv::Mat strip = img(cv::Rect(0, img.rows - stripH, img.cols, stripH));
    strip *= 0.4;                                                  // darken the bottom so the text is readable
    cv::Point pos((img.cols - sz.width) / 2, img.rows - base - 10);
    cv::putText(img, text, pos, font, scale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
}

int main() {
    cv::VideoCapture cap(0);   // open the webcam (0 = the built-in camera)
    cv::Mat frame;             // one picture from the camera
    cv::Mat display;           // the picture after all filters in the chain
    cv::Mat grey;              // greyscale copy of the frame for the face detector
    std::vector<char> chain;   // the filters to apply, in order (their keys)
    std::vector<cv::Rect> faces;      // faces found in the current frame
    std::vector<cv::Rect> lastFaces;  // most recent non-empty detection, reused briefly if the detector misses
    int missedFrames = 0;      // how many frames in a row the detector found nothing
    bool showFaces = false;    // 'f' toggles face boxes on/off
    DA2Network *daNet = nullptr;  // depth network, only loaded the first time a depth filter is used
    cv::Mat depth;             // depth from DA2 for this frame (CV_8UC1, 255 = near)
    cv::Mat depthSmooth;       // depth averaged over recent frames so it doesn't flicker

    while (true) {
        cap.read(frame);
        if (frame.empty()) break;

        // does the chain need faces or depth? (only compute them when needed, they are slow)
        bool needFaces = showFaces, needDepth = false;
        for (char k : chain) {
            if (k == 'o') needFaces = true;
            if (k == 'd' || k == 'z') needDepth = true;
        }
        if (needFaces) {                                  // detect on the ORIGINAL frame
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            detectFaces(grey, faces);
            if (!faces.empty()) { lastFaces = faces; missedFrames = 0; }
            else if (missedFrames < 5) { faces = lastFaces; missedFrames++; }  // hold the last boxes for up to 5 frames
        }
        if (needDepth) {                                  // depth from the ORIGINAL frame
            if (!daNet) daNet = new DA2Network("da2-code/model_fp16.onnx");
            daNet->set_input(frame, 256.0f / frame.rows);
            daNet->run_network(depth, frame.size());
            if (depthSmooth.size() != depth.size()) depth.copyTo(depthSmooth);
            else cv::addWeighted(depth, 0.5, depthSmooth, 0.5, 0, depthSmooth);  // running average: steadier
        }

        display = frame.clone();                          // start from the camera picture...
        for (char k : chain)                              // ...and apply every filter in order
            applyFilter(k, display, depthSmooth, faces);

        if (showFaces) drawBoxes(display, faces);
        drawChain(display, chain);                        // "Filter 1 + Filter 2 + ..." at the bottom
        cv::imshow("Filter Chain", display);

        char key = cv::waitKey(10);
        if (key == 'q') break;
        if (!filterName(key).empty()) chain.push_back(key);        // add a filter to the end of the chain
        if (key == 'u' && !chain.empty()) chain.pop_back();        // undo the last filter
        if (key == 'c') chain.clear();                             // back to the original video
        if (key == 'f') showFaces = !showFaces;
        if (key == 's') {
            cv::imwrite("data/chain.jpg", display);                // saved with the chain text on it
            std::cout << "Saved data/chain.jpg" << std::endl;
        }
    }

    delete daNet;
    return 0;
}
