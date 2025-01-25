//
//  OpenCVWrapper.mm
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 5/12/24.
//

#import "OpenCVWrapper.h"
#import <opencv2/opencv.hpp>

@implementation OpenCVWrapper

@synthesize delegate;

+ (nullable NSDictionary<NSString *, id> *)calibrateCameraAndGetDistortion:(NSString *)folderPath
                                                                     error:(NSError * _Nullable *)error {
    NSMutableDictionary<NSString *, id> *result = [NSMutableDictionary dictionary];
    
    // 加载图片路径
    NSError *fileError = nil;
    NSArray<NSString *> *imagePaths = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:folderPath error:&fileError];
    if (fileError) {
        if (error) *error = fileError;
        return nil;
    }

    if (imagePaths.count == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"com.reconstructionapp.opencv"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"No images found in folder"}];
        }
        return nil;
    }

    NSUInteger totalSteps = imagePaths.count + 1; // 包括标定步骤
    NSUInteger completedSteps = 0;

    // 准备数据结构
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Size imageSize;

    // 标定后的图像保存路径
    NSMutableArray<NSString *> *calibratedImagePaths = [NSMutableArray array];

    // 设置棋盘格大小 (行列数量)
    cv::Size patternSize(9, 6); // 示例：9x6 的棋盘格

    // 准备实际世界的棋盘格坐标
    std::vector<cv::Point3f> objectCorners;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objectCorners.emplace_back(j, i, 0);
        }
    }

    for (NSString *imagePath in imagePaths) {
        // 加载图像
        cv::Mat image = cv::imread([folderPath stringByAppendingPathComponent:imagePath].UTF8String, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            if (error) {
                *error = [NSError errorWithDomain:@"com.reconstructionapp.opencv"
                                             code:-2
                                         userInfo:@{NSLocalizedDescriptionKey: @"Failed to load image"}];
            }
            return nil;
        }

        // 检测角点
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // 精细化角点
            cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(objectCorners);

            // 可视化角点并保存标定后的图像
            cv::Mat annotatedImage;
            cv::cvtColor(image, annotatedImage, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(annotatedImage, patternSize, corners, found);

            // 生成保存路径
            NSString *calibratedImagePath = [folderPath stringByAppendingPathComponent:
                                             [NSString stringWithFormat:@"calibrated_%@", imagePath]];
            cv::imwrite([calibratedImagePath UTF8String], annotatedImage);

            // 添加路径到数组
            [calibratedImagePaths addObject:calibratedImagePath];
        }

        // 更新图像尺寸
        imageSize = image.size();

        // 更新进度
        completedSteps++;
        double progress = (double)completedSteps / totalSteps;
        [self updateProgress:progress];
    }

    // 进行标定
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    // 更新进度到完成状态
    completedSteps++;
    double finalProgress = (double)completedSteps / totalSteps;
    [self updateProgress:finalProgress];

    // 保存结果
    if (rms < 1.0) { // 假设 RMS 小于 1.0 表示成功
        NSMutableDictionary *coefficients = [NSMutableDictionary dictionary];
        coefficients[@"fx"] = @(cameraMatrix.at<double>(0, 0));
        coefficients[@"fy"] = @(cameraMatrix.at<double>(1, 1));
        coefficients[@"cx"] = @(cameraMatrix.at<double>(0, 2));
        coefficients[@"cy"] = @(cameraMatrix.at<double>(1, 2));
        coefficients[@"k1"] = @(distCoeffs.at<double>(0, 0));
        coefficients[@"k2"] = @(distCoeffs.at<double>(0, 1));
        coefficients[@"p1"] = @(distCoeffs.at<double>(0, 2));
        coefficients[@"p2"] = @(distCoeffs.at<double>(0, 3));
        coefficients[@"k3"] = @(distCoeffs.at<double>(0, 4));

        result[@"coefficients"] = coefficients;
        result[@"calibratedImages"] = calibratedImagePaths; // 添加标定后的图像路径
    } else {
        if (error) {
            *error = [NSError errorWithDomain:@"com.reconstructionapp.opencv"
                                         code:-3
                                     userInfo:@{NSLocalizedDescriptionKey: @"Calibration failed due to high RMS"}];
        }
        return nil;
    }

    return result;
}

+ (void)updateProgress:(double)progress {
    // 确保 delegate 存在并响应 calibrationProgress: 方法
    if (self.delegate && [self.delegate respondsToSelector:@selector(calibrationProgress:)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate calibrationProgress:progress];
        });
    }
}

@end
