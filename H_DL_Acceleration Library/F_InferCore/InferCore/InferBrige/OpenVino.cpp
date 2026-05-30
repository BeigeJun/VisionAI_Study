#include "pch.h"
#include "OpenVino.h"


bool OpenVino::bLoad(const std::string& strModelPath, const std::string& strDevice, const int nHeight, const int nWidth)
{
    try {
        m_Core.set_property(strDevice, ov::enable_profiling(true));

        auto ptrModel = m_Core.read_model(strModelPath + "/model.xml");
        ptrModel->reshape({ 1, 3, nHeight, nWidth });

        m_CompiledModel = m_Core.compile_model(ptrModel, strDevice);
        m_InferRequest = m_CompiledModel.create_infer_request();

        m_nNumClasses = (int)m_CompiledModel.output().get_partial_shape()[1].get_length();
        m_tensorInputTensor = m_InferRequest.get_input_tensor();
        m_shapeInputShape = m_InferRequest.get_input_tensor().get_shape();
        m_shapeOutputShape = m_InferRequest.get_output_tensor().get_shape();
		return true;
    }
	catch (...) { return false; }
}

std::vector<float> OpenVino::vecInfer(const std::vector<float>& vecInputData)
{
    std::memcpy(m_tensorInputTensor.data<float>(), vecInputData.data(), vecInputData.size() * sizeof(float));
    m_InferRequest.infer();
    auto tensor = m_InferRequest.get_output_tensor();
    float* ptr = tensor.data<float>();
    return std::vector<float>(ptr, ptr + tensor.get_size());
}

std::vector<size_t> OpenVino::vecReturnInputShape()
{
    return m_shapeInputShape;
}

std::vector<size_t> OpenVino::vecReturnOutputShape()
{
    return m_shapeOutputShape;
}

int OpenVino::nReturnClassNum()
{
    return m_nNumClasses;
}

