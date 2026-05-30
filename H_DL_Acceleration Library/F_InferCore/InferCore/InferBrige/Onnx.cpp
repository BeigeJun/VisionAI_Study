#include "pch.h"
#include "Onnx.h"

Onnx::Onnx() 
    : m_ortEnv(ORT_LOGGING_LEVEL_WARNING, "TransUnet"), 
      m_ortMemInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      m_nInputElemCount(0), 
      m_nNumClasses(0), 
      m_pcInput(nullptr), 
      m_pcOutput(nullptr) 
{
}

bool Onnx::bLoad(const std::string& strModelPath, const std::string& strDevice, const int nHeight, const int nWidth)
{
    try {
		std::string strFullPath = strModelPath+"/model.onnx";
        m_ortSessionOpts.SetIntraOpNumThreads(4);
        m_ortSessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        m_ortCudaOpts.device_id = 0;
        m_ortCudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        m_ortCudaOpts.gpu_mem_limit = SIZE_MAX;
        m_ortCudaOpts.arena_extend_strategy = 0;
        m_ortCudaOpts.do_copy_in_default_stream = 1;
        m_ortSessionOpts.AppendExecutionProvider_CUDA(m_ortCudaOpts);

        std::wstring wstrModelPath(strFullPath.begin(), strFullPath.end());
        mp_ortSession = std::make_unique<Ort::Session>(m_ortEnv, wstrModelPath.c_str(), m_ortSessionOpts);

        Ort::AllocatorWithDefaultOptions ortAllocator;

        m_strInputName = mp_ortSession->GetInputNameAllocated(0, ortAllocator).get();
        m_strOutputName = mp_ortSession->GetOutputNameAllocated(0, ortAllocator).get();

        m_pcInput = m_strInputName.c_str();
        m_pcOutput = m_strOutputName.c_str();

        m_nNumClasses = static_cast<int>(mp_ortSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);

        m_aInputShape = { 1, 3, static_cast<int64_t>(nHeight), static_cast<int64_t>(nWidth) };
        m_shapeOutputShape = mp_ortSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        m_nInputElemCount = 1 * 3 * nHeight * nWidth;
    }
    catch (...) {
        return false;
    }
}

std::vector<float> Onnx::vecInfer(const std::vector<float>& vecInputData)
{
    Ort::Value ortInputTensor = Ort::Value::CreateTensor<float>(
        m_ortMemInfo, const_cast<float*>(vecInputData.data()), m_nInputElemCount,
        m_aInputShape.data(), m_aInputShape.size());

    std::vector<Ort::Value> vecOrtOutputTensors =
        mp_ortSession->Run(Ort::RunOptions{ nullptr },
            &m_pcInput, &ortInputTensor, 1,
            &m_pcOutput, 1);

    const float* pfOutData = vecOrtOutputTensors[0].GetTensorData<float>();
    std::vector<int64_t> vecOutShape = vecOrtOutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t nSize = 1;
    for (auto dim : vecOutShape) nSize *= dim;

    return std::vector<float>(pfOutData, pfOutData + nSize);
}

std::vector<size_t> Onnx::vecReturnInputShape()
{
    return std::vector<size_t>(m_aInputShape.begin(), m_aInputShape.end());
}

std::vector<size_t> Onnx::vecReturnOutputShape()
{
    return std::vector<size_t>(m_shapeOutputShape.begin(), m_shapeOutputShape.end());
}

int Onnx::nReturnClassNum()
{
    return m_nNumClasses;
}
