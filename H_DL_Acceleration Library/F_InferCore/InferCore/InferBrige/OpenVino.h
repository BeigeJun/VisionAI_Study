#pragma once
#include "pch.h"
#include <openvino/openvino.hpp>
class OpenVino : public IInference
{
public:
    bool bLoad(const std::string& strModelPath, const std::string& strDevice = "CPU", const int nHeight = 900, const int nWidth = 900) override;
    std::vector<float> vecInfer(const std::vector<float>& vecInputData) override;
    std::vector<size_t> vecReturnInputShape() override; //const가 붙으면 읽기 전용으로 쓴다는뜻
    std::vector<size_t> vecReturnOutputShape() override;
    int nReturnClassNum() override;
private:
    ov::Core m_Core;
    ov::CompiledModel m_CompiledModel;
    ov::InferRequest m_InferRequest;
    ov::Tensor m_tensorInputTensor;
    ov::Shape m_shapeInputShape;
    ov::Shape m_shapeOutputShape;
    int m_nNumClasses;
    std::vector<float> m_vecImage;
};