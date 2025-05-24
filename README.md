# FaceIdentify Project

**Author:** Tang Nguyen-Tan  
**Email:** tangnt.1289@gmail.com  

This repository contains three main components for face identification and smart door locking:

- **FaceIdentify_CPP**: C++ application for face detection and embedding using TensorFlow Lite and OpenCV.
- **FaceIdentify_TFlite**: Python utilities and scripts for face detection and embedding with TensorFlow Lite models.
- **ZigBeeDoorLocker**: Embedded project for ZigBee-based smart door locking, designed for microcontrollers.

---

## Project Structure

```
face_identify/
├── FaceIdentify_CPP/      # C++ face detection & embedding app
├── FaceIdentify_TFlite/   # Python TFLite face detection & embedding
└── ZigBeeDoorLocker/      # Embedded ZigBee smart door lock
```

---

## 1. FaceIdentify_CPP

A C++ application for real-time face detection and embedding extraction using TensorFlow Lite and OpenCV.

- **Features**:  
    - MTCNN face detection  
    - FaceNet embedding extraction  
    - Configurable via `.conf` files  
    - Web UI for monitoring

- **Build & Run**:
    ```sh
    cd FaceIdentify_CPP
    cmake -B build
    cmake --build build -j$(nproc)
    ./build/FaceIdentify --config etc/camera_and_models.conf

    
- See [FaceIdentify_CPP/README.md](FaceIdentify_CPP/README.md) for full instructions.


![Demo](Video.gif)

---

## 2. FaceIdentify_TFlite

Python scripts and notebooks for face detection and embedding using TensorFlow Lite models.

- **Features**:  
    - MTCNN and FaceNet TFLite models  
    - Easy-to-edit model paths in `model_path.py`  
    - Jupyter notebook for testing

- **Usage**:
    ```python
    from facenet_tflite import MTCNNFaceNetTFlite
    # See face_detect_test.ipynb for examples
    ```

---

## 3. ZigBeeDoorLocker

Embedded firmware for a ZigBee-based smart door lock.

- **Features**:  
    - PlatformIO project structure  
    - Custom libraries in `lib/`  
    - Example code in `src/main.cpp`

- **Usage**:
    - Open with PlatformIO IDE or CLI
    - Build and upload to your microcontroller

---

## License

See [LICENSE](#license) below.

---

## References

- [OpenCV Documentation](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [PlatformIO Documentation](https://docs.platformio.org/)
- [MTCNN-OpenCV](https://github.com/egcode/mtcnn-opencv)

---

## LICENSE

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```