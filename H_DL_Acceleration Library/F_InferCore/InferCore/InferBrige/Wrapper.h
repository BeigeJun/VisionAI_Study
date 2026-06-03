#pragma once

extern "C" //맹글링 방지 -> 이름이 C 스타일대로 유지도되록함
{
    //__declspec(dllexport) DLL 외부에서 사용 가능하게 함
    __declspec(dllexport) void* Create(int nBackendType);
    __declspec(dllexport) bool  BLoad(void* pHandle, const char* pszModelPath, const char* pszDevice, int nHeight, int nWidth);
    __declspec(dllexport) void  Infer(void* pHandle, const float* pfInput, int nInputSize, float* pfOutput, int nOutputSize);
    __declspec(dllexport) void  ReturnInputShape(void* pHandle, size_t* pOut, int* pSize);
    __declspec(dllexport) void  ReturnOutputShape(void* pHandle, size_t* pOut, int* pSize);
    __declspec(dllexport) int   NReturnClassNum(void* pHandle);
    __declspec(dllexport) void  Destroy(void* pHandle);
}