#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <memory>

namespace fs = std::filesystem;

#ifdef OPENVINO_EXPORTS
#define DLL_EXPORT  __declspec(dllexport)
#else
#define DLL_EXPORT  __declspec(dllimport)
#endif

class DLL_EXPORT  TensorRT
{
public:
    TensorRT(std::string strEnginePath, std::string strDataPath, int nHeight, int nWidth);
    ~TensorRT();
    std::vector<float> vecInfer(const std::vector<float>& vecInputData);
    int nReturnClassNum() const;
    void Terminate();
private:
    class TrtLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity eSeverity, const char* pszMsg) noexcept override
        {
            if (eSeverity <= Severity::kWARNING)
                std::cerr << "[TRT] " << pszMsg << std::endl;
        }
    }m_Logger;

    std::unique_ptr<nvinfer1::IRuntime> m_upRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_upEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_upContext;
    cudaStream_t m_CudaStream;
	int m_nNumClasses;
    void* m_pDevInput;
    void* m_pDevOutput;
    size_t m_nInputBytes;
    size_t m_nOutputBytes;
    std::vector<float> m_vecfOutputHost;
};
