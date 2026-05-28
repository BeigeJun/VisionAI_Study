#pragma once
#include <iostream>
#include
#ifdef BRIGE_EXPORTS
#define DLL_EXPORT  __declspec(dllexport)
#else
#define DLL_EXPORT  __declspec(dllimport)
#endif

enum class FormetType
{
	ONNX,
	TENSORRT,
	OPENVINO,
	TORCH
};

class DLL_EXPORT Total_DLL
{
public:
	static std::unique_ptr<IInference> Create(FormetType type);
};