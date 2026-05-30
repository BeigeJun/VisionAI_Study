// pch.h: 미리 컴파일된 헤더 파일입니다.
// 아래 나열된 파일은 한 번만 컴파일되었으며, 향후 빌드에 대한 빌드 성능을 향상합니다.
// 코드 컴파일 및 여러 코드 검색 기능을 포함하여 IntelliSense 성능에도 영향을 미칩니다.
// 그러나 여기에 나열된 파일은 빌드 간 업데이트되는 경우 모두 다시 컴파일됩니다.
// 여기에 자주 업데이트할 파일을 추가하지 마세요. 그러면 성능이 저하됩니다.

#ifndef PCH_H
#define PCH_H
#define NOMINMAX

#define USE_OPENVINO
#define USE_ONNX
#define USE_TENSORRT

#include "framework.h"
#include "IInference.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

#define CUDA_LIB    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/"
#define TRT_LIB     "D:/5.Lib_Files/1.TensorRT/TensorRT-10.10.0.31/lib/"
#define ONNX_LIB    "D:/5.Lib_Files/5.Onnx/onnxruntime-win-x64-gpu-1.26.0/lib/"
#define OV_LIB      "D:/5.Lib_Files/3.OpenVino/runtime/lib/intel64/"
#define TORCH_LIB   "D:/5.Lib_Files/4.Pytorch/"

#pragma comment(lib, CUDA_LIB "cudart.lib")

#ifdef _DEBUG
#pragma comment(lib, TRT_LIB "nvinfer_10.lib")
#pragma comment(lib, TRT_LIB "nvinfer_plugin_10.lib")
#pragma comment(lib, TRT_LIB "nvonnxparser_10.lib")
#pragma comment(lib, ONNX_LIB "onnxruntime.lib")
#pragma comment(lib, OV_LIB "Debug/openvinod.lib")
#pragma comment(lib, TORCH_LIB "Debug/libtorch/lib/torch.lib")
#pragma comment(lib, TORCH_LIB "Debug/libtorch/lib/torch_cpu.lib")
#pragma comment(lib, TORCH_LIB "Debug/libtorch/lib/torch_cuda.lib")
#pragma comment(lib, TORCH_LIB "Debug/libtorch/lib/c10.lib")
#pragma comment(lib, TORCH_LIB "Debug/libtorch/lib/c10_cuda.lib")

#else
#pragma comment(lib, TRT_LIB "nvinfer_10.lib")
#pragma comment(lib, TRT_LIB "nvinfer_plugin_10.lib")
#pragma comment(lib, TRT_LIB "nvonnxparser_10.lib")
#pragma comment(lib, ONNX_LIB "onnxruntime.lib")
#pragma comment(lib, OV_LIB "Release/openvino.lib")
#pragma comment(lib, TORCH_LIB "Release/libtorch/lib/torch.lib")
#pragma comment(lib, TORCH_LIB "Release/libtorch/lib/torch_cpu.lib")
#pragma comment(lib, TORCH_LIB "Release/libtorch/lib/torch_cuda.lib")
#pragma comment(lib, TORCH_LIB "Release/libtorch/lib/c10.lib")
#pragma comment(lib, TORCH_LIB "Release/libtorch/lib/c10_cuda.lib")
#endif

#endif //PCH_H
