#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem; // 使用文件系统命名空间

// 加载文件夹中的所有 JPEG 图片路径
std::vector<std::string> loadImagePaths(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpeg") {
                imagePaths.push_back(entry.path().string());
            }
        }
        // 按文件名排序
        std::sort(imagePaths.begin(), imagePaths.end());
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    return imagePaths;
}

int main() {
    // 棋盘格设置
    const int CHESSBOARD_ROWS = 6; // 行数
    const int CHESSBOARD_COLS = 8; // 列数
    const float SQUARE_SIZE = 25.0; // 每个方格的边长（单位：毫米）

    // 加载图片
    std::string folderPath = "../pictures"; // 替换为实际文件夹路径
    std::vector<std::string> imagePaths = loadImagePaths(folderPath);

    if (imagePaths.empty()) {
        std::cerr << "No JPEG images found in the folder: " << folderPath << std::endl;
        return -1;
    }

    // 输出加载的文件路径
    std::cout << "Loaded images:" << std::endl;
    for (const auto& path : imagePaths) {
        std::cout << path << std::endl;
    }

    // 其余代码保持不变（处理图片、标定相机等）
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < CHESSBOARD_ROWS; ++i) {
        for (int j = 0; j < CHESSBOARD_COLS; ++j) {
            objectPoints.emplace_back(j * SQUARE_SIZE, i * SQUARE_SIZE, 0.0f);
        }
    }

    std::vector<std::vector<cv::Point3f>> objectPointsList; // 世界坐标系
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

            cv::drawChessboardCorners(image, cv::Size(CHESSBOARD_COLS, CHESSBOARD_ROWS), imagePoints, found);
            cv::imshow("Chessboard Detection", image);
            cv::waitKey(100);
        } else {
            std::cerr << "Chessboard corners not found in image: " << imagePath << std::endl;
        }
    }

    cv::destroyAllWindows();

    if (imagePointsList.size() < 3) {
        std::cerr << "Not enough valid images for calibration!" << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Size imageSize = cv::imread(imagePaths[0]).size();

    double reprojectionError = cv::calibrateCamera(objectPointsList, imagePointsList, imageSize,
                                                   cameraMatrix, distCoeffs, rvecs, tvecs);

    std::ofstream csvFile("distortion_coefficients.csv");
    if (csvFile.is_open()) {
        csvFile << "k1,k2,p1,p2,k3\n"; // 添加表头
        for (int i = 0; i < distCoeffs.cols; ++i) {
            csvFile << distCoeffs.at<double>(0, i);
            if (i < distCoeffs.cols - 1) {
                csvFile << ","; // 用逗号分隔
            }
        }
        csvFile << "\n"; // 换行
        csvFile.close();
        std::cout << "Distortion coefficients saved to distortion_coefficients.csv" << std::endl;
    } else {
        std::cerr << "Failed to open file for writing distortion coefficients." << std::endl;
    }

    return 0;
}