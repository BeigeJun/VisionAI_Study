#include "pch.h"
#include "Total_DLL.h"


std::unique_ptr<IInference> Total_DLL::Create(BackendType type)
{
    switch (type)
    {
#ifdef USE_TENSORRT
    case BackendType::TensorRT:
        return std::make_unique<TensorRT>();
#endif

#ifdef USE_ONNX
    case BackendType::ONNX:
        return std::make_unique<Onnx>();
#endif

#ifdef USE_OPENVINO
    case BackendType::OpenVINO:
        return std::make_unique<OpenVino>();
#endif

    default:
        throw std::invalid_argument("지원하지 않는 백엔드입니다.");
        return nullptr;
    }
}