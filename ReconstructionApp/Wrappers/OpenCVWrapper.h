//
//  OpenCVWrapper.h
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 5/12/24.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/**
 OpenCVWrapper 用于封装相机校准功能，支持角点检测和畸变参数计算。
 */
@protocol OpenCVWrapperDelegate <NSObject>

/**
 当校准进度更新时调用该方法。

 @param progress 当前进度，范围为 0.0 ~ 1.0。
 */
- (void)calibrationProgress:(double)progress;

@end

@interface OpenCVWrapper : NSObject

/// 代理，用于报告校准进度
@property (nonatomic, weak) id<OpenCVWrapperDelegate> delegate;

/**
 检测棋盘格角点并返回畸变参数和标定后的图像。

 @param folderPath 包含棋盘格图片的文件夹路径。
 @param error 如果发生错误，则返回错误信息。
 @return 一个字典，包含内参矩阵、畸变系数和校准后的图片路径。
 */
+ (nullable NSDictionary<NSString *, id> *)calibrateCameraAndGetDistortion:(NSString *)folderPath
                                                                     error:(NSError * _Nullable *)error;

/**
 将校准结果保存到指定路径。

 @param result 校准结果，包含内参矩阵和畸变系数。
 @param outputPath 保存的目标路径。
 @param error 如果发生错误，则返回错误信息。
 */
+ (void)saveCalibrationResult:(NSDictionary<NSString *, id> *)result
                        toPath:(NSString *)outputPath
                         error:(NSError * _Nullable *)error;

@end

NS_ASSUME_NONNULL_END

