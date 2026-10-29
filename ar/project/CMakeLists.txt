cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED PROJECT_NAME)
    set(PROJECT_NAME "main")
endif()

project(${PROJECT_NAME})

# 设置目录路径
set(HOME_PATH "/home/elf")
set(PROJECT_LIBS "${CMAKE_SOURCE_DIR}/lib")
set(PROJECT_INC_DIRS "${CMAKE_SOURCE_DIR}/include")
set(PROJECT_SRC_DIRS "${CMAKE_SOURCE_DIR}/src")
set(PROJECT_3RD_DIRS "${CMAKE_SOURCE_DIR}/3rdparty")

find_package(OpenCV REQUIRED)

aux_source_directory(${PROJECT_SRC_DIRS} SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

#链接库文件
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    m
    glfw
    GL
)

#添加头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_INC_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)