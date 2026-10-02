# CS 5330 Project 1: Video Special Effects

Akashdeep Gangatkar Madhusudhan

Time Travel Days: 2 Days

Extensions: 
- Canny filter and canny with no blur.
- Stacking multiple or same filters on the same frame.

## Requirements
- C++17 compiler (tested with Apple clang on macOS)
- OpenCV 5 (Homebrew `opencv`). `faceDetect.cpp` includes `opencv2/xobjdetect.hpp`, which is where
  OpenCV 5 moved `CascadeClassifier`. On OpenCV 4, remove that include line.
- ONNX Runtime (Homebrew `onnxruntime`), for the Depth Anything V2 network
- A webcam for `vidDisplay` and `chainDisplay`

## Build
```
make          # builds all programs with -O2
make clean    # removes the built programs
```
Run every program **from this folder**, because the programs load `haarcascade_frontalface_alt2.xml`,
`da2-code/model_fp16.onnx` and the images in `data/` and `output/` with relative paths.

## Programs

### `./imgDisplay` (Task 1)
Shows `data/image1.jpg`. Press any key to close.

### `./vidDisplay` (Tasks 2-12, Extension 1)
Live webcam with one effect at a time. Pressing an effect key again goes back to colour.

| Key | Effect | Key | Effect |
|---|---|---|---|
| `g` | OpenCV greyscale | `l` | Blur + quantize |
| `h` | Custom greyscale | `d` | Depth map (DA2) |
| `p` | Sepia | `z` | Depth portrait mode |
| `v` | Sepia + vignette | `r` | Red colour pop |
| `b` | 5x5 Gaussian blur | `t` | Cartoon |
| `x` | Sobel X | `o` | Face spotlight |
| `y` | Sobel Y | `e` | Canny edges (Extension 1, from scratch) |
| `m` | Gradient magnitude | `f` | Face boxes on/off (works with any effect) |

`s` saves the current picture to `data/saved.jpg`, `q` quits.
The depth model loads the first time `d` or `z` is pressed, which takes a moment.

### `./chainDisplay` (Extension 2)
Stacks filters: each effect key (same keys as above) **adds** that filter to a chain, and every filter
is applied to the output of the previous one. The current chain is shown at the bottom of the video,
e.g. `Sepia + Blur + Canny`. The same filter can be added several times.

- `u` = remove the last filter, `c` = clear the chain
- `f` = face boxes on/off, `s` = save to `data/chain.jpg`, `q` = quit

### `./timeBlur [image]` (Task 6)
Times `blur5x5_1` against `blur5x5_2` on `data/image1.jpg` (or the given image) and saves the results
to `data/blur1.jpg` and `data/blur2.jpg`.

### `./output/applyFilters`
Applies every filter to `output/original image.jpg` and saves each result in `output/`, named after the
filter (plus OpenCV's greyscale and Canny for comparison).

## Files
- `filter.cpp`, `filter.h`: all filters written for this project
- `faceDetect.cpp`, `faceDetect.h`, `haarcascade_frontalface_alt2.xml`: provided face detection
- `da2-code/DA2Network.hpp`, `da2-code/model_fp16.onnx`: provided Depth Anything V2 wrapper and model
- `report/report.pdf`: project report
