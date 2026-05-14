#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "D:/5.Lib_Files/3.OpenVino/runtime/include/openvino/openvino.hpp"
#include "D:/5.Lib_Files/2.opencv/build/include/opencv2/opencv.hpp"
#include "D:/5.Lib_Files/2.opencv/build/include/opencv2/core/utils/logger.hpp"


#ifdef _DEBUG
#pragma comment(lib, "D:/5.Lib_Files/3.OpenVino/runtime/lib/intel64/Debug/openvinod.lib")
#pragma comment(lib, "D:/5.Lib_Files/2.opencv/build/x64/vc16/bin/opencv_world4100d.lib")

#else
#pragma comment(lib, "D:/5.Lib_Files/3.OpenVino/runtime/lib/intel64/Release/openvino.lib")
#pragma comment(lib, "D:/5.Lib_Files/2.opencv/build/x64/vc16/bin/opencv_world4100.lib")
#endif

namespace fs = std::filesystem;