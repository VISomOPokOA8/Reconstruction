cmake_minimum_required(VERSION 3.10)

# 项目名称
project(CameraCalibration)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 添加源文件
add_executable(camera_calibration main.cpp)

# 链接 OpenCV 库
target_link_libraries(camera_calibration ${OpenCV_LIBS})

# 确保支持 C++17 的文件系统库（针对某些编译器可能需要额外标志）
if(MSVC)
    target_compile_options(camera_calibration PRIVATE /std:c++17)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(camera_calibration PRIVATE -std=c++17)
endif()

# 包含路径（如果需要，可以添加自定义路径）
include_directories(${OpenCV_INCLUDE_DIRS})