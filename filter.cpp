#include "filter.h"
#include <algorithm>

// Custom greyscale: take the largest of the three colour values (B, G, R) and copy it
// into all three channels. Saturated colours (bright red, blue...) come out bright,
// unlike cvtColor which weights green heavily and makes blue look dark.
int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    dst.create(src.size(), src.type());                   // make dst the same size/type as src (3 channels)

    for (int i = 0; i < src.rows; i++) {                           // go through every row
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);          // pointer to the start of row i in src
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);                // pointer to the start of row i in dst
        for (int j = 0; j < src.cols; j++) {                       // go through every pixel in the row
            uchar b = srcRow[j][0], g = srcRow[j][1], r = srcRow[j][2];  // OpenCV order is B, G, R
            uchar v = std::max({b, g, r});                         // brightest channel
            dstRow[j] = cv::Vec3b(v, v, v);                        // same value in all three channels = grey
        }
    }
    return 0;
}

// Sepia tone: every new channel is a mix of all three ORIGINAL channels.
// If vignette is true, pixels are also darkened the further they are from the centre.
int sepia(cv::Mat &src, cv::Mat &dst, bool vignette) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    dst.create(src.size(), src.type());                   // make dst the same size/type as src

    float cx = src.cols / 2.0f, cy = src.rows / 2.0f;     // centre of the image
    float maxDist2 = cx * cx + cy * cy;                   // squared distance from centre to a corner

    for (int i = 0; i < src.rows; i++) {                           // go through every row
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);          // pointer to the start of row i in src
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);                // pointer to the start of row i in dst
        for (int j = 0; j < src.cols; j++) {                       // go through every pixel in the row
            float b = srcRow[j][0], g = srcRow[j][1], r = srcRow[j][2];  // read the original values first

            float newB = 0.272f * r + 0.534f * g + 0.131f * b;     // blue coefficients for R, G, B
            float newG = 0.349f * r + 0.686f * g + 0.168f * b;     // green coefficients
            float newR = 0.393f * r + 0.769f * g + 0.189f * b;     // red coefficients

            if (vignette) {                                        // darken towards the edges
                float dx = j - cx, dy = i - cy;
                float d2 = (dx * dx + dy * dy) / maxDist2;         // 0 at the centre, 1 at a corner
                float factor = 1.0f - 0.7f * d2;                   // corners keep 30% of their brightness
                newB *= factor; newG *= factor; newR *= factor;
            }

            dstRow[j][0] = (uchar)std::min(newB, 255.0f);          // clamp to 255 so values don't wrap around
            dstRow[j][1] = (uchar)std::min(newG, 255.0f);
            dstRow[j][2] = (uchar)std::min(newR, 255.0f);
        }
    }
    return 0;
}

// Naive 5x5 Gaussian blur: for every pixel, multiply the 5x5 neighbourhood by the
// kernel below, add it up and divide by the kernel total (100). Uses .at<> for every read.
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    static const int kernel[5][5] = {    // integer approximation of a Gaussian, sums to 100
        {1, 2,  4, 2, 1},
        {2, 4,  8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4,  8, 4, 2},
        {1, 2,  4, 2, 1}
    };

    src.copyTo(dst);                     // start with a copy so the outer 2 rows/cols keep src values

    for (int i = 2; i < src.rows - 2; i++) {              // skip the outer two rows
        for (int j = 2; j < src.cols - 2; j++) {          // skip the outer two columns
            for (int c = 0; c < 3; c++) {                 // blur B, G, R separately
                int sum = 0;
                for (int ki = -2; ki <= 2; ki++)          // 5x5 neighbourhood around (i, j)
                    for (int kj = -2; kj <= 2; kj++)
                        sum += kernel[ki + 2][kj + 2] * src.at<cv::Vec3b>(i + ki, j + kj)[c];
                dst.at<cv::Vec3b>(i, j)[c] = (uchar)(sum / 100);   // normalise by the kernel total
            }
        }
    }
    return 0;
}

// Faster 5x5 Gaussian blur: the 5x5 kernel equals [1 2 4 2 1] (vertical) times [1 2 4 2 1]
// (horizontal), so do a horizontal pass then a vertical pass (10 multiplies per pixel instead
// of 25), and walk rows with pointers instead of calling .at<> for every pixel.
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    src.copyTo(dst);                                     // outer 2 rows/cols keep src values (non-zero)
    cv::Mat tmp(src.size(), CV_16UC3, cv::Scalar(0));    // horizontal sums, up to 10*255 = 2550 so 16 bits is enough

    // pass 1: horizontal [1 2 4 2 1] on every row
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *s = src.ptr<cv::Vec3b>(i);       // row i of src
        cv::Vec3w *t = tmp.ptr<cv::Vec3w>(i);             // row i of tmp
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                t[j][c] = s[j - 2][c] + 2 * s[j - 1][c] + 4 * s[j][c] + 2 * s[j + 1][c] + s[j + 2][c];
            }
        }
    }

    // pass 2: vertical [1 2 4 2 1] on the horizontal sums, then divide by 10*10 = 100
    for (int i = 2; i < src.rows - 2; i++) {
        const cv::Vec3w *t0 = tmp.ptr<cv::Vec3w>(i - 2);  // the five tmp rows around row i
        const cv::Vec3w *t1 = tmp.ptr<cv::Vec3w>(i - 1);
        const cv::Vec3w *t2 = tmp.ptr<cv::Vec3w>(i);
        const cv::Vec3w *t3 = tmp.ptr<cv::Vec3w>(i + 1);
        const cv::Vec3w *t4 = tmp.ptr<cv::Vec3w>(i + 2);
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);             // row i of dst
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                int sum = t0[j][c] + 2 * t1[j][c] + 4 * t2[j][c] + 2 * t3[j][c] + t4[j][c];
                d[j][c] = (uchar)(sum / 100);
            }
        }
    }
    return 0;
}

// 3x3 Sobel X (positive when the right side is brighter), done as two 1x3 passes:
//   vertical [1 2 1] smoothing, then horizontal [-1 0 1] difference, divided by 4 so
//   the result stays in [-255, 255].  Edge pixels reuse the nearest valid row/col.
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    cv::Mat tmp(src.size(), CV_16SC3);                    // vertical sums, up to 4*255 = 1020
    dst.create(src.size(), CV_16SC3);                     // signed output, can be negative

    // pass 1: vertical [1 2 1]
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *up = src.ptr<cv::Vec3b>(std::max(i - 1, 0));             // row above (clamped)
        const cv::Vec3b *mid = src.ptr<cv::Vec3b>(i);                              // this row
        const cv::Vec3b *down = src.ptr<cv::Vec3b>(std::min(i + 1, src.rows - 1)); // row below (clamped)
        cv::Vec3s *t = tmp.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++)
            for (int c = 0; c < 3; c++)
                t[j][c] = up[j][c] + 2 * mid[j][c] + down[j][c];
    }

    // pass 2: horizontal [-1 0 1]  (right minus left = positive right)
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3s *t = tmp.ptr<cv::Vec3s>(i);
        cv::Vec3s *d = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            int left = std::max(j - 1, 0), right = std::min(j + 1, src.cols - 1);  // clamped neighbours
            for (int c = 0; c < 3; c++)
                d[j][c] = (short)((t[right][c] - t[left][c]) / 4);
        }
    }
    return 0;
}

// 3x3 Sobel Y (positive when the top is brighter), done as two 1x3 passes:
//   horizontal [1 2 1] smoothing, then vertical [1 0 -1] difference (above minus below),
//   divided by 4 so the result stays in [-255, 255].
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;   // only handle normal 3-channel colour images

    cv::Mat tmp(src.size(), CV_16SC3);                    // horizontal sums, up to 4*255 = 1020
    dst.create(src.size(), CV_16SC3);                     // signed output, can be negative

    // pass 1: horizontal [1 2 1]
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *s = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *t = tmp.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            int left = std::max(j - 1, 0), right = std::min(j + 1, src.cols - 1);  // clamped neighbours
            for (int c = 0; c < 3; c++)
                t[j][c] = s[left][c] + 2 * s[j][c] + s[right][c];
        }
    }

    // pass 2: vertical [1 0 -1]  (row above minus row below = positive up)
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3s *up = tmp.ptr<cv::Vec3s>(std::max(i - 1, 0));              // row above (clamped)
        const cv::Vec3s *down = tmp.ptr<cv::Vec3s>(std::min(i + 1, src.rows - 1));  // row below (clamped)
        cv::Vec3s *d = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++)
            for (int c = 0; c < 3; c++)
                d[j][c] = (short)((up[j][c] - down[j][c]) / 4);
    }
    return 0;
}

// Gradient magnitude: for each pixel and channel, I = sqrt(sx^2 + sy^2).
// sx and sy are CV_16SC3 Sobel outputs; dst is CV_8UC3 so it can be shown directly.
// The largest possible value is sqrt(2)*255 ~ 361, so results are clamped to 255.
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3 || sx.size() != sy.size())
        return -1;                                        // both inputs must be matching signed-short colour images

    dst.create(sx.size(), CV_8UC3);                       // 8-bit colour output for display

    for (int i = 0; i < sx.rows; i++) {
        const cv::Vec3s *x = sx.ptr<cv::Vec3s>(i);        // row i of Sobel X
        const cv::Vec3s *y = sy.ptr<cv::Vec3s>(i);        // row i of Sobel Y
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);             // row i of dst
        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {                 // each colour channel separately
                float gx = x[j][c], gy = y[j][c];
                float m = std::sqrt(gx * gx + gy * gy);   // Euclidean length of the gradient
                d[j][c] = (uchar)std::min(m, 255.0f);     // clamp so it fits in a uchar
            }
        }
    }
    return 0;
}

// Blur, then quantize each colour channel into 'levels' values (cartoon/poster look).
// Bucket size b = 255 / levels; each value x becomes (x / b) * b.
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.type() != CV_8UC3 || levels < 1) return -1;

    blur5x5_2(src, dst);                                  // reuse our fast blur; dst is now the blurred image

    int b = 255 / levels;                                 // size of one bucket
    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);             // row i of dst (quantized in place)
        for (int j = 0; j < dst.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int xt = d[j][c] / b;                     // which bucket the value falls in
                xt = std::min(xt, levels - 1);            // integer b can give one extra bucket at the top; fold it in
                d[j][c] = (uchar)(xt * b);                // snap to the bottom of that bucket
            }
        }
    }
    return 0;
}

// Depth "portrait mode": near pixels stay sharp and in colour, far pixels get a heavy blur
// and fade towards grey.  depth is CV_8UC1 from DA2, same size as src, 255 = near, 0 = far.
// Pixels with depth above 'hi' are fully sharp, below 'lo' fully background, smooth blend between.
int depthFocus(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int lo, int hi) {
    if (src.empty() || src.type() != CV_8UC3 || depth.type() != CV_8UC1 ||
        depth.size() != src.size() || hi <= lo) return -1;

    // strong background blur: shrink to 1/4, blur twice with our own 5x5 blur, grow back
    cv::Mat small, smallBlur, bg;
    cv::resize(src, small, cv::Size(), 0.25, 0.25);
    blur5x5_2(small, smallBlur);
    blur5x5_2(smallBlur, small);
    cv::resize(small, bg, src.size());

    dst.create(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *s = src.ptr<cv::Vec3b>(i);       // original row
        const cv::Vec3b *b = bg.ptr<cv::Vec3b>(i);        // blurred row
        const uchar *z = depth.ptr<uchar>(i);             // depth row
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            float t = (z[j] - lo) / (float)(hi - lo);     // 0 = background, 1 = subject
            t = std::min(std::max(t, 0.0f), 1.0f);
            float w = t * t * (3 - 2 * t);                // smoothstep so the edge isn't harsh

            float grey = 0.114f * b[j][0] + 0.587f * b[j][1] + 0.299f * b[j][2];  // grey of the blurred pixel
            for (int c = 0; c < 3; c++) {
                float back = 0.4f * b[j][c] + 0.6f * grey;                        // blurred and mostly desaturated
                d[j][c] = (uchar)(w * s[j][c] + (1 - w) * back);                  // mix subject and background
            }
        }
    }
    return 0;
}

// ---------------- Task 12 effects ----------------

// Colour pop (pixel-wise): strongly red pixels keep their colour, everything else turns grey.
// "Redness" = R minus the larger of G and B; a smooth ramp between 40 and 80 avoids hard cut-outs.
int colourPop(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *s = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int b = s[j][0], g = s[j][1], r = s[j][2];
            float redness = r - std::max(g, b);                   // high only for saturated reds
            float t = std::min(std::max((redness - 40) / 40.0f, 0.0f), 1.0f);
            float w = t * t * (3 - 2 * t);                        // smoothstep: 0 = grey, 1 = keep colour
            float grey = 0.114f * b + 0.587f * g + 0.299f * r;    // same weights as cvtColor
            for (int c = 0; c < 3; c++)
                d[j][c] = (uchar)(w * s[j][c] + (1 - w) * grey);
        }
    }
    return 0;
}

// Cartoon / video abstraction, following Winnemoeller, Olsen & Gooch (2006):
//   1. convert to CIELab (L in [0,100]) and apply 4 bilateral filter passes (sigma_d = 3, sigma_r = 4.25)
//   2. DoG edges on L after 2 passes: sigma_e = 1, second blur 1.6*sigma_e, tau = 0.98, phi_e = 2
//   3. soft luminance quantization, sharpness phi_q set from the L gradient (our own Sobel + magnitude)
//   4. multiply quantized L by the edge map, convert back to BGR
// Runs at half resolution for speed.
int cartoon(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.type() != CV_8UC3 || levels < 2) return -1;

    // 1. half size, float Lab, iterated bilateral filter
    cv::Mat small, lab, tmp, edgeL;
    cv::resize(src, small, cv::Size(), 0.5, 0.5);
    small.convertTo(lab, CV_32FC3, 1.0 / 255);            // float BGR in [0,1] so cvtColor gives L in [0,100]
    cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
    for (int k = 0; k < 4; k++) {
        cv::bilateralFilter(lab, tmp, 9, 4.25, 3);        // edge-preserving smoothing (can't run in place)
        cv::swap(lab, tmp);
        if (k == 1) cv::extractChannel(lab, edgeL, 0);    // edges use the image after n_e = 2 passes
    }

    // 2. difference-of-Gaussians edges: D = 1 on flat areas, dips towards 0 on lines
    cv::Mat g1, g2;
    cv::GaussianBlur(edgeL, g1, cv::Size(0, 0), 1.0);
    cv::GaussianBlur(edgeL, g2, cv::Size(0, 0), 1.6);

    // 3. L-gradient for the quantization sharpness, using our own Sobel/magnitude on an 8-bit copy of L
    cv::Mat L, L8, L8c, sx, sy, mag;
    cv::extractChannel(lab, L, 0);
    L.convertTo(L8, CV_8U, 2.55);                         // [0,100] -> [0,255]
    cv::cvtColor(L8, L8c, cv::COLOR_GRAY2BGR);            // our filters expect 3 channels
    sobelX3x3(L8c, sx);
    sobelY3x3(L8c, sy);
    magnitude(sx, sy, mag);

    const float dq = 100.0f / levels;                     // bin width in L units
    for (int i = 0; i < L.rows; i++) {
        float *l = L.ptr<float>(i);
        const float *a = g1.ptr<float>(i);
        const float *b = g2.ptr<float>(i);
        const cv::Vec3b *m = mag.ptr<cv::Vec3b>(i);
        for (int j = 0; j < L.cols; j++) {
            // DoG edge value (paper eq. 4)
            float diff = a[j] - 0.98f * b[j];
            float D = diff > 0 ? 1.0f : 1.0f + std::tanh(2.0f * diff);

            // gradient in L units per pixel (our Sobel is /4, i.e. about 2x the per-pixel slope)
            float grad = m[j][0] / 2.55f / 2.0f;
            float phi = 3.0f + std::min(grad, 2.0f) / 2.0f * (14.0f - 3.0f);   // [0,2] -> [3,14]

            // soft quantization (paper eq. 6): snap to nearest bin boundary, tanh step between bins
            float qn = std::round(l[j] / dq) * dq;
            float Q = qn + dq / 2 * std::tanh(phi * (l[j] - qn));

            l[j] = std::min(std::max(Q * D, 0.0f), 100.0f);
        }
    }

    // 4. put L back, convert to BGR, back to full size
    cv::insertChannel(L, lab, 0);
    cv::cvtColor(lab, lab, cv::COLOR_Lab2BGR);
    lab.convertTo(small, CV_8UC3, 255);
    cv::resize(small, dst, src.size());
    return 0;
}

// Face spotlight: faces stay in colour inside a soft oval, the rest becomes our custom greyscale
// (greyscale()) dimmed to 60%.  Ignores boxes narrower than minWidth, like drawBoxes.
int faceSpotlight(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst, int minWidth) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    cv::Mat grey;
    greyscale(src, grey);                                 // reuse our Task 4 greyscale for the background

    dst.create(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *s = src.ptr<cv::Vec3b>(i);
        const cv::Vec3b *g = grey.ptr<cv::Vec3b>(i);
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            float w = 0;                                  // 1 = inside a face oval, 0 = background
            for (const cv::Rect &f : faces) {
                if (f.width <= minWidth) continue;
                float cx = f.x + f.width / 2.0f, cy = f.y + f.height / 2.0f;
                float rx = 0.75f * f.width, ry = 0.95f * f.height;   // oval a bit bigger than the box (hair, chin)
                float dx = (j - cx) / rx, dy = (i - cy) / ry;
                float r2 = dx * dx + dy * dy;             // < 1 inside the oval
                float t = std::min(std::max((1.3f - r2) / 0.6f, 0.0f), 1.0f);  // soft edge from r2 = 0.7 to 1.3
                w = std::max(w, t * t * (3 - 2 * t));
            }
            for (int c = 0; c < 3; c++)
                d[j][c] = (uchar)(w * s[j][c] + (1 - w) * 0.6f * g[j][c]);
        }
    }
    return 0;
}

// Canny edge detector, written from scratch on top of our earlier filters.
// Output is CV_8UC3: white (255) on thin 1-pixel edges, black everywhere else.
// 'low' and 'high' are in the same units as magnitude() (0..~361):
//   above high = strong edge, between low and high = weak edge (kept only if it touches a strong one).
int canny(cv::Mat &src, cv::Mat &dst, int low, int high) {
    if (src.empty() || src.type() != CV_8UC3 || low > high) return -1;

    // step 1: blur to remove noise, then turn into grey (same weights as cvtColor)
    cv::Mat blurred, grey(src.size(), CV_8UC3);
    blur5x5_2(src, blurred);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b *b = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *g = grey.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            uchar v = (uchar)(0.114f * b[j][0] + 0.587f * b[j][1] + 0.299f * b[j][2]);
            g[j] = cv::Vec3b(v, v, v);                    // 3 equal channels so our Sobel filters accept it
        }
    }

    // step 2: Sobel X and Y (all three channels are equal, so we only read channel 0 below)
    cv::Mat sx, sy;
    sobelX3x3(grey, sx);
    sobelY3x3(grey, sy);

    // step 3: gradient magnitude, and the direction rounded to one of 4 sectors
    //   sector 0 = horizontal gradient (vertical edge), 1 = 45 deg, 2 = vertical gradient, 3 = 135 deg
    cv::Mat mag(src.size(), CV_32FC1), dir(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3s *x = sx.ptr<cv::Vec3s>(i);
        const cv::Vec3s *y = sy.ptr<cv::Vec3s>(i);
        float *m = mag.ptr<float>(i);
        uchar *d = dir.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            float gx = x[j][0], gy = -y[j][0];            // our Sobel Y is positive UP; flip so +gy points down the rows
            m[j] = std::sqrt(gx * gx + gy * gy);
            float a = std::atan2(gy, gx) * 180.0f / (float)CV_PI;   // -180..180 degrees
            if (a < 0) a += 180;                          // a line has no front/back, so 0..180 is enough
            if (a < 22.5f || a >= 157.5f) d[j] = 0;
            else if (a < 67.5f) d[j] = 1;
            else if (a < 112.5f) d[j] = 2;
            else d[j] = 3;
        }
    }

    // step 4: non-maximum suppression (thinning)
    //   keep a pixel only if it is at least as strong as its two neighbours ACROSS the edge
    //   (along the gradient direction); this turns thick edges into 1-pixel lines
    const int dRow[4] = {0, 1, 1, 1};                     // neighbour offset for each sector (the other one is the opposite)
    const int dCol[4] = {1, 1, 0, -1};
    cv::Mat thin(src.size(), CV_32FC1, cv::Scalar(0));    // border pixels stay 0
    for (int i = 1; i < src.rows - 1; i++) {
        const uchar *d = dir.ptr<uchar>(i);
        float *t = thin.ptr<float>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            float m = mag.at<float>(i, j);
            int k = d[j];
            float n1 = mag.at<float>(i + dRow[k], j + dCol[k]);   // neighbour on one side
            float n2 = mag.at<float>(i - dRow[k], j - dCol[k]);   // neighbour on the other side
            if (m >= n1 && m >= n2) t[j] = m;
        }
    }

    // step 5: double threshold -> 2 = strong, 1 = weak, 0 = not an edge
    cv::Mat edge(src.size(), CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> stack;                         // strong pixels still to grow from (used in step 6)
    for (int i = 0; i < src.rows; i++) {
        const float *t = thin.ptr<float>(i);
        uchar *e = edge.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            if (t[j] >= high) { e[j] = 2; stack.push_back(cv::Point(j, i)); }
            else if (t[j] >= low) e[j] = 1;
        }
    }

    // step 6: hysteresis -- starting from every strong pixel, turn touching weak pixels into strong ones,
    //   and keep going from those; weak pixels never reached are dropped
    while (!stack.empty()) {
        cv::Point p = stack.back();
        stack.pop_back();
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {            // the 8 neighbours
                int r = p.y + di, c = p.x + dj;
                if (r < 0 || r >= src.rows || c < 0 || c >= src.cols) continue;
                if (edge.at<uchar>(r, c) == 1) {
                    edge.at<uchar>(r, c) = 2;
                    stack.push_back(cv::Point(c, r));
                }
            }
        }
    }

    // write the result: strong edges white, everything else black
    dst.create(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const uchar *e = edge.ptr<uchar>(i);
        cv::Vec3b *d = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            uchar v = (e[j] == 2) ? 255 : 0;
            d[j] = cv::Vec3b(v, v, v);
        }
    }
    return 0;
}
