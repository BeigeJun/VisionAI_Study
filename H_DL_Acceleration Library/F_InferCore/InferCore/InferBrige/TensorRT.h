#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class  TensorRT : public IInference
{
public:
    ~TensorRT();
    bool bLoad(const std::string& strModelPath, const std::string& strDevice = "GPU", const int nHeight = 900, const int nWidth = 900) override;
    std::vector<float> vecInfer(const std::vector<float>& vecInputData) override;
    std::vector<size_t> vecReturnInputShape() override;
    std::vector<size_t> vecReturnOutputShape() override;
    int nReturnClassNum() override;
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
    std::vector<int64_t> m_vecInputShape;
    std::vector<int64_t> m_vecOutputShape;
    std::vector<float> m_vecfOutputHost;
};
