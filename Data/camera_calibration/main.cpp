#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// 加载文件夹中的所有 JPEG 图片路径
std::vector<std::string> loadImagePaths(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpeg") {
                imagePaths.push_back(entry.path().string());
            }
        }
        std::sort(imagePaths.begin(), imagePaths.end());
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    return imagePaths;
}

// 返回畸变参数的函数
std::tuple<double, double, double, double, double> calibrateCameraAndGetDistortion(const std::string& folderPath) {
    // 棋盘格设置
    const int CHESSBOARD_ROWS = 6;
    const int CHESSBOARD_COLS = 8;
    const float SQUARE_SIZE = 25.0;

    // 加载图片
    std::vector<std::string> imagePaths = loadImagePaths(folderPath);

    if (imagePaths.empty()) {
        throw std::runtime_error("No JPEG images found in the folder: " + folderPath);
    }

    // 准备棋盘格世界坐标
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < CHESSBOARD_ROWS; ++i) {
        for (int j = 0; j < CHESSBOARD_COLS; ++j) {
            objectPoints.emplace_back(j * SQUARE_SIZE, i * SQUARE_SIZE, 0.0f);
        }
    }

    std::vector<std::vector<cv::Point3f>> objectPointsList;  // 世界坐标系
    std::vector<std::vector<cv::Point2f>> imagePointsList;  // 图像坐标系

    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }

        std::vector<cv::Point2f> imagePoints;
        bool found = cv::findChessboardCorners(image, cv::Size(CHESSBOARD_COLS, CHESSBOARD_ROWS), imagePoints);
        if (found) {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, imagePoints, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            imagePointsList.push_back(imagePoints);
            objectPointsList.push_back(objectPoints);
        } else {
            std::cerr << "Chessboard corners not found in image: " << imagePath << std::endl;
        }
    }

    if (imagePointsList.size() < 3) {
        throw std::runtime_error("Not enough valid images for calibration!");
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Size imageSize = cv::imread(imagePaths[0]).size();

    double reprojectionError = cv::calibrateCamera(objectPointsList, imagePointsList, imageSize,
                                                   cameraMatrix, distCoeffs, rvecs, tvecs);

    // 提取畸变参数
    if (distCoeffs.cols < 5) {
        throw std::runtime_error("Insufficient distortion coefficients returned!");
    }

    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(0, 1);
    double p1 = distCoeffs.at<double>(0, 2);
    double p2 = distCoeffs.at<double>(0, 3);
    double k3 = distCoeffs.at<double>(0, 4);

    return {k1, k2, p1, p2, k3};
}

int main() {
    try {
        std::string folderPath = "../pictures";  // 替换为实际文件夹路径
        auto [k1, k2, p1, p2, k3] = calibrateCameraAndGetDistortion(folderPath);

        std::cout << "Distortion coefficients:" << std::endl;
        std::cout << "k1 = " << k1 << std::endl;
        std::cout << "k2 = " << k2 << std::endl;
        std::cout << "p1 = " << p1 << std::endl;
        std::cout << "p2 = " << p2 << std::endl;
        std::cout << "k3 = " << k3 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}