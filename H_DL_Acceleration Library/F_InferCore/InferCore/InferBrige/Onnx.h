#pragma once
#include "pch.h"
#include <onnxruntime_cxx_api.h>
class Onnx : public IInference
{
public:
    Onnx();
    bool bLoad(const std::string& strModelPath, const std::string& strDevice = "GPU", const int nHeight = 900, const int nWidth = 900) override;
    std::vector<float> vecInfer(const std::vector<float>& vecInputData) override;
    std::vector<size_t> vecReturnInputShape() override;
    std::vector<size_t> vecReturnOutputShape() override;
    int nReturnClassNum() override;

private:
    Ort::Env m_ortEnv;
    Ort::SessionOptions m_ortSessionOpts;
    OrtCUDAProviderOptions m_ortCudaOpts{};
    std::unique_ptr<Ort::Session> mp_ortSession;
    Ort::MemoryInfo m_ortMemInfo;
    size_t m_nInputElemCount;

    std::string m_strInputName;
    std::string m_strOutputName;

    std::vector<int64_t> m_aInputShape;
    std::vector<int64_t> m_shapeOutputShape;

    const char* m_pcInput;
    const char* m_pcOutput;

    int m_nNumClasses;
};