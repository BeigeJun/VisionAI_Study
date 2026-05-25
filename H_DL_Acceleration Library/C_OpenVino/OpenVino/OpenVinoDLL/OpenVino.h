#pragma once
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

// DLL export/import 매크로
#ifdef OPENVINO_EXPORTS
#define DLL_EXPORT  __declspec(dllexport)
#else
#define DLL_EXPORT  __declspec(dllimport)
#endif

class DLL_EXPORT  OpenVino
{
public:
    OpenVino(const std::string& strModelPath, const std::string& strDevice = "CPU", const int nHeight = 900, const int nWidth = 900);
    std::vector<float> vecInfer(const std::vector<float>& vecInputData);
    std::vector<size_t> vecReturnInputShape() const; //const가 붙으면 읽기 전용으로 쓴다는뜻
    std::vector<size_t> vecReturnOutputShape() const;
    int nReturnClassNum() const;
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