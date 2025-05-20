# FaceIdentify

## Author
**Tang Nguyen-Tan**  

**Email:** tangnt.1289@gmail.com  

## Environment
**Ubuntu 24.04 LTS**

**Python 3.12** (for building TensorFlow processes)

## Update and Install Prerequisites
```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y cmake g++ wget zip unzip make git libboost-all-dev libgtk2.0-dev pkg-config

sudo apt install -y \
    build-essential libavutil-dev libswresample-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev libcanberra-gtk3-dev \
    libv4l-dev v4l-utils \
    libxvidcore-dev libx264-dev libx265-dev \
    libopenblas-dev libatlas-base-dev liblapack-dev gfortran

```

## Download and Install OpenCV

### Steps:
1. **Download and unpack sources:**
    ```sh
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
    ```

2. **Create build directory and configure:**
    ```sh
    mkdir -p build && cd build
    cmake -D CMAKE_CXX_STANDARD=23 -D ENABLE_CXX11=ON -D BUILD_SHARED_LIBS=ON -D CMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
    ```

3. **Fix compatibility issue:**
    - Open the file `opencv-4.x/modules/gapi/include/opencv2/gapi/util/variant.hpp`.
    - At line 76, replace:
      ```cpp
      using Memory = typename std::aligned_storage<S, A>::type[1];
      ```
      with:
      ```cpp
      using Memory = std::aligned_storage_t<S, A>[1];
      ```

4. **Build and install:**
    ```sh
    cmake --build . -- -j$(nproc)
    sudo make install -j$(nproc)
    cd .. && rm -rf build open*
    ```

## Download and Install tensorflow 

### Steps:

```sh

sudo apt install -y python3.12-venv python3-dev python3-pip python3-numpy python3-wheel
python3 -m venv ./.tf_venv
source  .tf_venv/bin/activate
pip3 install six numpy wheel
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r2.19
cat .bazelversion  # Check the Bazel version must be installed
cd ..

```

1. **(Optional) Install CUDA and cuDNN driver**
    If GPU card is available, the CUDA driver should be installed to greatly boost both training and inference speed.

2. **Install Bazel:**
    ```sh
    sudo apt install -y apt-transport-https curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
    sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    sudo apt update && sudo apt install -y bazel-6.5.0
    sudo ln -s /usr/bin/bazel-6.5.0 /usr/bin/bazel
    ```

3. **Install Protobuf:**
    ```sh
    sudo apt-get install -y autoconf automake libtool
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git submodule update --init --recursive
    bazel build :protoc :protobuf
    sudo cp bazel-bin/protoc /usr/local/bin
    cd .. && rm -rf protobuf
    ```

4. **Install Eigen:**
    ```sh
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip eigen-3.4.0.zip && cd eigen-3.4.0
    mkdir build && cd build
    make -j$(nproc)
    sudo make install
    cd ../../ && rm -rf eigen*
    ```

5. **Install tensorflow or tensorflow lite:**
    ```sh
    cd tensorflow
    # If GPU card is available you can enable ROCm or CUDA support for TensorFlow.
    ./configure

    # (Optional) Build tensorflow c++ with bazel (you can change the number of cores used with jobs flag)
    # --config=monolithic is important, so tf works with opencv (for cv::imread)
    # --config=cuda need to for NVIDIA GPU

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both \
        --copt=-msse4.2 --config=monolithic --jobs $(nproc) //tensorflow:libtensorflow_cc.so
    
    # clean the last build results
    bazel clean

    # Build tensorflow lite c++ with bazel 
    bazel build -c opt --copt=-march=native --config=monolithic --jobs $(nproc)  //tensorflow/lite:libtensorflowlite.so

    # Install tensorflow lite c++
    sudo cp bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/local/lib/
    sudo mkdir -p /usr/local/include/tensorflow
    sudo cp -r tensorflow/lite /usr/local/include/tensorflow/
    sudo mkdir -p /usr/local/include/tensorflow/compiler/mlir/
    sudo cp -r tensorflow/compiler/mlir/lite/ /usr/local/include/tensorflow/compiler/mlir/

    # Copy third party 
    sudo cp -rL bazel-bin/external/flatbuffers/_virtual_includes/flatbuffers/flatbuffers /usr/local/include/
    sudo cp -L bazel-bin/external/flatbuffers/_virtual_includes/runtime_cc/flatbuffers/* /usr/local/include/flatbuffers
    sudo cp -rL bazel-bin/external/FP16/_virtual_includes/FP16 /usr/local/include/
    sudo cp -rL bazel-bin/external/FXdiv/_virtual_includes/FXdiv /usr/local/include/
    sudo cp -rL bazel-bin/external/cpuinfo/_virtual_includes/cpuinfo /usr/local/include/
    sudo cp -rL bazel-bin/external/pthreadpool/_virtual_includes/pthreadpool /usr/local/include/

    cd .. && rm -rf tensorflow
    rm -rf ~/.cache/bazel

    ```

6. **(Optional) Install Abseil:**
    ```sh
    
    git clone https://github.com/abseil/abseil-cpp.git
    cd abseil-cpp

    cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/install
    cmake --build build -j$(nproc)
    sudo cmake --install build

    # Copy to your project
    # Header in: abseil-cpp/install/include/absl/...
    # Lib in: abseil-cpp/install/lib/...

    cd .. && rm -rf abseil-cpp

    ```

## Build this project

```sh

git clone https://github.com/tangnguyendeveloper/face_identify.git
cd face_identify/FaceIdentify_CPP
cmake -B build
cmake --build build -j$(nproc)

# Change the etc/camera_and_models.conf if you need
# run project
./build/FaceIdentify etc/camera_and_models.conf

```



References:
[OpenCV Documentation](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

[Build LiteRT](https://ai.google.dev/edge/litert/build/arm)

[Google Facenet implementation for live face recognition in C++ using TensorFlow, OpenCV, and dlib](https://github.com/nwesem/facenet_cpp_tensorflow/tree/master)

[C++ MTCNN inferencing with only OpenCV](https://github.com/egcode/mtcnn-opencv)
