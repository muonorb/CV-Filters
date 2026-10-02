#include <opencv2/opencv.hpp>
#include<iostream>
#include "filter.h"            // our own image filters (filter.cpp)
#include "faceDetect.h"        // provided face detection code (faceDetect.cpp)
#include "da2-code/DA2Network.hpp"  // provided Depth Anything V2 wrapper (needs ONNX Runtime)

int main() {
    cv::VideoCapture cap(0);   // open the webcam (0 = the built-in camera)
    cv::Mat frame;             // empty image that will hold one picture from the camera
    cv::Mat display;           // the picture we actually show (colour or grey)
    cv::Mat sobelOut;          // raw signed Sobel result (CV_16SC3), kept separate from what we show
    cv::Mat sobelY;            // second Sobel result, needed for the gradient magnitude
    cv::Mat grey;              // greyscale copy of the frame for the face detector
    std::vector<cv::Rect> faces;  // boxes around the faces found in the current frame
    bool showFaces = false;    // 'f' toggles face boxes on/off (works on top of any effect)
    std::vector<cv::Rect> lastFaces;  // most recent non-empty detection, reused briefly if the detector misses
    int missedFrames = 0;      // how many frames in a row the detector found nothing
    DA2Network *daNet = nullptr;  // depth network, only loaded the first time a depth mode is used
    cv::Mat depth;             // depth from DA2 for this frame (CV_8UC1, 255 = near)
    cv::Mat depthSmooth;       // depth averaged over recent frames so it doesn't flicker
    char mode = 'c';           // remembers the current effect: 'c' = colour, 'g' = OpenCV grey, 'h' = custom grey,
                               // 'p' = sepia, 'v' = sepia + vignette, 'b' = blur,
                               // 'x' = Sobel X, 'y' = Sobel Y, 'm' = gradient magnitude,
                               // 'l' = blur + quantize,
                               // 'd' = depth map, 'z' = depth portrait mode,
                               // 'r' = red colour pop, 't' = cartoon, 'o' = face spotlight,
                               // 'e' = Canny edges

    while (true) {                           // keep repeating until we break out
        cap.read(frame);                     // grab the next picture from the webcam into 'frame'
        if (frame.empty()) break;            // if no picture came in (camera failed), stop the loop
        if (showFaces || mode == 'o') {      // find faces first: spotlight mode needs them, 'f' draws them later
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);  // the detector needs a greyscale image
            detectFaces(grey, faces);        // fill 'faces' with one rectangle per face
            if (!faces.empty()) { lastFaces = faces; missedFrames = 0; }
            else if (missedFrames < 5) { faces = lastFaces; missedFrames++; }  // hold the last boxes for up to 5 frames
        }
        if (mode == 'g') cv::cvtColor(frame, display, cv::COLOR_BGR2GRAY);  // OpenCV grey: 1-channel weighted grey
        else if (mode == 'h') greyscale(frame, display);                     // custom grey from filter.cpp
        else if (mode == 'p') sepia(frame, display);                         // sepia tone
        else if (mode == 'v') sepia(frame, display, true);                   // sepia tone with dark edges
        else if (mode == 'b') blur5x5_2(frame, display);                     // 5x5 Gaussian blur (fast version)
        else if (mode == 'x') {                                              // Sobel X: shows vertical edges
            sobelX3x3(frame, sobelOut);
            cv::convertScaleAbs(sobelOut, display);                          // |value| as 8-bit so imshow can draw it
        }
        else if (mode == 'y') {                                              // Sobel Y: shows horizontal edges
            sobelY3x3(frame, sobelOut);
            cv::convertScaleAbs(sobelOut, display);
        }
        else if (mode == 'm') {                                              // gradient magnitude: all edges
            sobelX3x3(frame, sobelOut);
            sobelY3x3(frame, sobelY);
            magnitude(sobelOut, sobelY, display);                            // already 8-bit, no conversion needed
        }
        else if (mode == 'l') blurQuantize(frame, display, 10);              // blur, then 10 levels per channel
        else if (mode == 'd' || mode == 'z') {                               // depth-based modes
            if (!daNet) daNet = new DA2Network("da2-code/model_fp16.onnx");  // load the model once (takes a moment)
            daNet->set_input(frame, 256.0f / frame.rows);                    // network sees a 256-pixel-tall image (speed vs quality)
            daNet->run_network(depth, frame.size());                         // depth resized back to the frame size
            if (depthSmooth.size() != depth.size()) depth.copyTo(depthSmooth);
            else cv::addWeighted(depth, 0.5, depthSmooth, 0.5, 0, depthSmooth);  // running average: steadier than one frame
            if (mode == 'd') cv::applyColorMap(depthSmooth, display, cv::COLORMAP_INFERNO);  // bright/yellow = near
            else depthFocus(frame, depthSmooth, display);                    // sharp subject, blurred grey background
        }
        else if (mode == 'r') colourPop(frame, display);                     // reds stay colour, rest grey
        else if (mode == 't') cartoon(frame, display, 10);                   // Winnemoeller cartoon, 10 luminance bins
        else if (mode == 'o') faceSpotlight(frame, faces, display);          // faces in colour, rest dim grey
        else if (mode == 'e') canny(frame, display, 5, 50);                         // from-scratch Canny edges
        else display = frame;                // colour mode: show the frame as-is
        if (showFaces) drawBoxes(display, faces);  // draw the face rectangles onto the displayed image
        cv::imshow("Video", display);        // show the picture in a window called "Video"
        char key = cv::waitKey(10);          // wait 10 ms and remember which key was pressed (if any)
        if (key == 'q') break;               // 'q' = stop the loop
        if (key == 'g' || key == 'h' || key == 'p' || key == 'v' || key == 'b' ||
            key == 'x' || key == 'y' || key == 'm' || key == 'l' ||
            key == 'd' || key == 'z' || key == 'r' || key == 't' || key == 'o' ||
            key == 'e')  // effect keys toggle on/off
            mode = (mode == key) ? 'c' : key;  // pressing the same key again goes back to colour
        if (key == 'f') showFaces = !showFaces;  // 'f' = toggle face detection on/off
        if (key == 's') {                    // 's' = save the current picture
            cv::imwrite("data/saved.jpg", display);  // write what's on screen to data/saved.jpg
            std::cout << "Saved data/saved.jpg" << std::endl;  // tell the user it worked
        }
    }

    delete daNet;              // free the depth network (safe even if it was never loaded)
    return 0;
}
