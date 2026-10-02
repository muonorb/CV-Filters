# CS 5330 Project 1 - build all programs with:  make
# Needs OpenCV (pkg-config name opencv5 or opencv4) and ONNX Runtime (libonnxruntime).

CXX      = clang++
OPENCV  := $(shell pkg-config --exists opencv5 && echo opencv5 || echo opencv4)
CXXFLAGS = -std=c++17 -O2 $(shell pkg-config --cflags $(OPENCV) libonnxruntime)
LDLIBS   = $(shell pkg-config --libs $(OPENCV) libonnxruntime)

FILTERS  = filter.cpp faceDetect.cpp
HEADERS  = filter.h faceDetect.h da2-code/DA2Network.hpp

all: vidDisplay chainDisplay imgDisplay timeBlur output/applyFilters

vidDisplay: vidDisplay.cpp $(FILTERS) $(HEADERS)
	$(CXX) $(CXXFLAGS) vidDisplay.cpp $(FILTERS) -o $@ $(LDLIBS)

chainDisplay: chainDisplay.cpp $(FILTERS) $(HEADERS)
	$(CXX) $(CXXFLAGS) chainDisplay.cpp $(FILTERS) -o $@ $(LDLIBS)

imgDisplay: imgDisplay.cpp
	$(CXX) $(CXXFLAGS) imgDisplay.cpp -o $@ $(LDLIBS)

timeBlur: timeBlur.cpp $(FILTERS) $(HEADERS)
	$(CXX) $(CXXFLAGS) timeBlur.cpp $(FILTERS) -o $@ $(LDLIBS)

output/applyFilters: output/applyFilters.cpp $(FILTERS) $(HEADERS)
	$(CXX) $(CXXFLAGS) output/applyFilters.cpp $(FILTERS) -o $@ $(LDLIBS)

clean:
	rm -rf vidDisplay chainDisplay imgDisplay timeBlur output/applyFilters *.dSYM

.PHONY: all clean
