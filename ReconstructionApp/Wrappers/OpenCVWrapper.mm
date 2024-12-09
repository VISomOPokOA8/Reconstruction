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

    // 模拟加载图片路径
    NSArray<NSString *> *imagePaths = @[@"image1.jpg", @"image2.jpg", @"image3.jpg"]; // 示例文件
    NSUInteger totalSteps = imagePaths.count + 1; // 包括其他计算步骤
    NSUInteger completedSteps = 0;

    for (NSString *imagePath in imagePaths) {
        // 模拟处理单张图片（例如角点检测）
        BOOL success = [self processImageAtPath:imagePath];
        completedSteps++;

        // 更新进度
        double progress = (double)completedSteps / totalSteps;
        [self updateProgress:progress];

        if (!success) {
            if (error) {
                *error = [NSError errorWithDomain:@"com.reconstructionapp.opencv"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Image processing failed"}];
            }
            return nil;
        }
    }

    // 模拟校准其他步骤
    completedSteps++;
    double progress = (double)completedSteps / totalSteps;
    [self updateProgress:progress];

    // 模拟返回校准结果
    result[@"coefficients"] = @{@"k1": @0.1, @"k2": @0.05}; // 示例系数
    result[@"calibratedImages"] = @[@"calibrated1.jpg", @"calibrated2.jpg", @"calibrated3.jpg"]; // 示例图像

    return result;
}

+ (BOOL)processImageAtPath:(NSString *)imagePath {
    // 模拟图片处理（例如角点检测）
    [NSThread sleepForTimeInterval:1.0]; // 模拟耗时操作
    return YES;
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
