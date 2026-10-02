#include<opencv2/opencv.hpp>   
#include<iostream>             

int main(){                    
    cv::Mat image = cv::imread("data/image1.jpg", cv::IMREAD_COLOR);  // open the picture file and store it in 'image' (as a color image)
    cv::imshow("Image", image);  // open a window called "Image" and show the picture in it
    cv::waitKey(0)!='q';         // wait until any key is pressed (the !='q' part does nothing - any key closes it)
    cv::destroyAllWindows();     // close the image window
    return 0;                    

}
