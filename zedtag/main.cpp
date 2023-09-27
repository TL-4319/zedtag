// Tuan Luong
// 9-20-2023
//

#include <sl/Camera.hpp>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <random>
#include <iostream>
#include <cstring>
#include <vector>
#include <list>
#include <sys/time.h>

// OpenCV library for easy access to USB camera and drawing of images
// on screen
#include "opencv2/opencv.hpp"

// OpenCV dep
#include <opencv2/cvconfig.h>
#include <opencv2/core/hal/interface.h>

// April tags detector and various families that can be selected by command line option
#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag16h5.h"
#include "AprilTags/Tag25h7.h"
#include "AprilTags/Tag25h9.h"
#include "AprilTags/Tag36h9.h"
#include "AprilTags/Tag36h11.h"

const char* windowName = "apriltags_demo";

int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

// utility function to provide current system time (used below in
// determining frame rate at which images are being processed)
double tic() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return ((double)t.tv_sec + ((double)t.tv_usec)/1000000.);
}

#include <cmath>

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

/**
 * Convert rotation matrix to Euler angles
 */
void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}

void print_detection(AprilTags::TagDetection& detection) {
    cout << "  Id: " << detection.id
         << " (Hamming: " << detection.hammingDistance << ")";

    // recovering the relative pose of a tag:

    // NOTE: for this to be accurate, it is necessary to use the
    // actual camera parameters here as well as the actual tag size
    // (m_fx, m_fy, m_px, m_py, m_tagSize)

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    detection.getRelativeTranslationRotation(0.166d, 526.195d, 526.105d, 0.0001d, -0.0005d,
                                             translation, rotation);

    Eigen::Matrix3d F;
    F <<
      1, 0,  0,
      0,  -1,  0,
      0,  0,  1;
    Eigen::Matrix3d fixed_rot = F*rotation;
    double yaw, pitch, roll;
    wRo_to_euler(fixed_rot, yaw, pitch, roll);

    cout << "  distance=" << translation.norm()
         << "m, x=" << translation(0)
         << ", y=" << translation(1)
         << ", z=" << translation(2)
         << ", yaw=" << yaw * 180 / PI
         << ", pitch=" << pitch * 180 / PI
         << ", roll=" << roll * 180 / PI
         << endl;

    // Also note that for SLAM/multi-view application it is better to
    // use reprojection error of corner points, because the noise in
    // this relative pose is very non-Gaussian; see iSAM source code
    // for suitable factors.
}

void processImage(cv::Mat& image, cv::Mat& image_gray, AprilTags::TagDetector* tagDetector) {
    // alternative way is to grab, then retrieve; allows for
    // multiple grab when processing below frame rate - v4l keeps a
    // number of frames buffered, which can lead to significant lag
    //      m_cap.grab();
    //      m_cap.retrieve(image);

    // detect April tags (requires a gray scale image)
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    std::vector<AprilTags::TagDetection> detections = tagDetector->extractTags(image_gray);

    // print out each detection
    std::cout << detections.size() << " tags detected:" << endl;
    for (int i=0; i<detections.size(); i++) {
      print_detection(detections[i]);
    }

    // show the current image including any detections
    if (true) {
      for (int i=0; i<detections.size(); i++) {
        // also highlight in the image
        detections[i].draw(image);
      }
      cv::imshow(windowName, image); // OpenCV call
    }
}

int main () {
  // Setup camera
  sl::Camera zed;
	sl::InitParameters init_param;
	init_param.camera_resolution = sl::RESOLUTION::HD720;
	init_param.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
	init_param.camera_fps = 15;
	init_param.coordinate_units = sl::UNIT::METER;
	init_param.depth_minimum_distance = 0.4;
	init_param.depth_maximum_distance = 4.0;
	int mat_len, width, height;
	// TODO ADD MORE RESOLUTION OPTIONS
	if (init_param.camera_resolution == sl::RESOLUTION::HD1080){
		width = 1920;
		height = 1080;
	}
	if (init_param.camera_resolution == sl::RESOLUTION::HD720) {
		width = 1280;
		height = 720;
	}
	else if (init_param.camera_resolution == sl::RESOLUTION::VGA){
		width = 672;
		height = 376;
	}
	mat_len = width * height;

  sl::Resolution new_image_size(width, height);

  sl::Mat slimg(width, height, sl::MAT_TYPE::U8_C4);


	auto returned_state = zed.open(init_param);
	if(returned_state != sl::ERROR_CODE::SUCCESS) {
		std::cout << "ERROR: " << returned_state << ", exit program.\n";
		return 1;
	}

    // Setup opencv/ detector
    cv::Mat cvimg = slMat2cvMat(slimg);
    cv::Mat image_gray;
    int cam_width = 1280;
    int cam_height = 720;
    double tagSize = 0.166;
    double cam_fx = 526.195f;
    double cam_fy = 526.105;
    double cam_px = 0.0001;
    double cam_py = -0.0005;
    AprilTags::TagDetector* tagDetector = new AprilTags::TagDetector(AprilTags::tagCodes36h11);
    cv::namedWindow(windowName, 1);

    // Loop
    while(true){
        if (zed.grab() == sl::ERROR_CODE::SUCCESS){
            zed.retrieveImage (slimg, sl::VIEW::LEFT);
            processImage(cvimg, image_gray, tagDetector);
            if (cv::waitKey(1) >= 0) break;
        }
    }

}