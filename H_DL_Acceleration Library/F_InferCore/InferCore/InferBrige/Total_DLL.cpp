#include "pch.h"
#include "Total_DLL.h"


std::unique_ptr<IInference> InferFactory::Create(BackendType type) {
    switch (type) {
    case BackendType::TensorRT:  return std::make_unique<TRTInference>();
    case BackendType::ONNX:      return std::make_unique<OnnxInference>();
    case BackendType::PyTorch:   return std::make_unique<TorchInference>();
    case BackendType::OpenVINO:  return std::make_unique<OVInference>();
    default: return nullptr;
    }
}