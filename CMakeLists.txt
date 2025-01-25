cmake_minimum_required(VERSION 3.30)
project(Algo)

set(CMAKE_CXX_STANDARD 20)


set(OpenCV_DIR ./3rdpart/opencv/x64/vc15/staticlib)
set(ncnn_DIR ./3rdpart/ncnn/lib/cmake/ncnn)
set(ncnn_INCLUDE_DIRS ./3rdpart/ncnn/include)

find_package(OpenCV REQUIRED) # 找到opencv库
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(ncnn REQUIRED)    # 找到ncnn库
include_directories(${ncnn_INCLUDE_DIRS})

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

include_directories(src)

add_library(Algo STATIC src/library.cpp
        src/Crop.cpp
        src/MobileCLIPViT.cpp
)
add_executable(CropTest src/demo_crop.cpp)
add_executable(ViTTest src/demo_vit.cpp)
add_executable(PicDBTest src/demo_picdb.cpp)

target_link_libraries(Algo ncnn)
target_link_libraries(Algo ${OpenCV_LIBS})

target_link_libraries(CropTest PRIVATE Algo)
target_link_libraries(ViTTest PRIVATE Algo)
target_link_libraries(PicDBTest PRIVATE Algo)


