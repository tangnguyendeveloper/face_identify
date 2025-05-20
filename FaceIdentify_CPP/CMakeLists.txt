cmake_minimum_required(VERSION 3.10)
project(FaceIdentify)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# Set the output directory for the executable
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Set the output directory for the library
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Set the output directory for the archive
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# Add source files
file(GLOB_RECURSE SOURCES src/*.cpp app/*.hpp)
set(MAIN_APP app/main.cpp)

# Add header files for IDEs (optional, not needed for build)
file(GLOB_RECURSE HEADERS src/*.hpp src/*.h)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_path(EIGEN3_INCLUDE_DIR Eigen/Dense HINTS /usr/local/include/eigen3)

if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Eigen3")
endif()

# Find TensorFlow Lite
find_library(TFLITE_LIB tensorflowlite HINTS /usr/local/lib REQUIRED)

# Create executable
add_executable(FaceIdentify ${MAIN_APP} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(FaceIdentify PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/app
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/include             # FlatBuffers, Abseil nếu cài ở đây
    /usr/local/include/tensorflow # tensorflow/lite/...
)

# Link OpenCV libraries
target_link_libraries(FaceIdentify PRIVATE ${OpenCV_LIBS} ${TFLITE_LIB})
