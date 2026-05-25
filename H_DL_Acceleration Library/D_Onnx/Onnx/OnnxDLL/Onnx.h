#pragma once
#include <onnxruntime_cxx_api.h>
#include <Windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

#ifdef ONNX_EXPORTS
#define DLL_EXPORT  __declspec(dllexport)
#else
#define DLL_EXPORT  __declspec(dllimport)
#endif

class DLL_EXPORT Onnx
{
public:
    Onnx(const std::string& strModelPath, const int nHeight = 512, const int nWidth = 512);
    std::vector<float> vecInfer(const std::vector<float>& vecInputData);
    std::array<int64_t, 4> vecReturnInputShape() const;
    std::vector<int64_t> vecReturnOutputShape() const;
    int nReturnClassNum() const;

private:
    Ort::Env m_ortEnv;
    Ort::SessionOptions m_ortSessionOpts;
    OrtCUDAProviderOptions m_ortCudaOpts{};
	std::unique_ptr<Ort::Session> mp_ortSession;
    Ort::MemoryInfo m_ortMemInfo;
    size_t m_nInputElemCount;

    std::string m_strInputName;
    std::string m_strOutputName;

    std::array<int64_t, 4> m_aInputShape;
    std::vector<int64_t> m_shapeOutputShape;

    const char* mp_cInput;
    const char* mp_cOutput;

	int m_nNumClasses;
};