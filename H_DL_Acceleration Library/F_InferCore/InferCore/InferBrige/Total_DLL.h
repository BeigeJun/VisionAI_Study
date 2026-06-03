#pragma once
#include "pch.h"
#include <iostream>
#include "IInference.h"
#include "OpenVino.h"
#include "Onnx.h"
#include "TensorRT.h"

#ifdef BRIGE_EXPORTS
#define DLL_EXPORT  __declspec(dllexport)
#else
#define DLL_EXPORT  __declspec(dllimport)
#endif

enum class BackendType {
    ONNX,
    OpenVINO,
    TensorRT
};
class DLL_EXPORT Total_DLL
{
public:
	static std::unique_ptr<IInference> Create(BackendType type);
};