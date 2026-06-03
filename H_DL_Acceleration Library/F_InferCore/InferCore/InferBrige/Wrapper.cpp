#include "pch.h"
#include "Wrapper.h"
#include "Total_DLL.h"

extern "C"
{
    void* Create(int nBackendType)
    {
        return Total_DLL::Create(static_cast<BackendType>(nBackendType)).release();
    }

    bool BLoad(void* pHandle, const char* pszModelPath, const char* pszDevice,
        int nHeight, int nWidth)
    {
        return reinterpret_cast<IInference*>(pHandle)->bLoad(pszModelPath, pszDevice, nHeight, nWidth);
    }

    void Infer(void* pHandle, const float* pfInput, int nInputSize, float* pfOutput, int nOutputSize)
    {
        auto* p = reinterpret_cast<IInference*>(pHandle);
        std::vector<float> vecInput(pfInput, pfInput + nInputSize);
        auto vecOutput = p->vecInfer(vecInput);
        memcpy(pfOutput, vecOutput.data(), nOutputSize * sizeof(float));
    }

    void ReturnInputShape(void* pHandle, size_t* pOut, int* pSize)
    {
        auto vec = reinterpret_cast<IInference*>(pHandle)->vecReturnInputShape();
        *pSize = static_cast<int>(vec.size());
        memcpy(pOut, vec.data(), vec.size() * sizeof(size_t));
    }

    void ReturnOutputShape(void* pHandle, size_t* pOut, int* pSize)
    {
        auto vec = reinterpret_cast<IInference*>(pHandle)->vecReturnOutputShape();
        *pSize = static_cast<int>(vec.size());
        memcpy(pOut, vec.data(), vec.size() * sizeof(size_t));
    }

    int NReturnClassNum(void* pHandle)
    {
        return reinterpret_cast<IInference*>(pHandle)->nReturnClassNum();
    }

    void Destroy(void* pHandle)
    {
        delete reinterpret_cast<IInference*>(pHandle);
    }
}

//reinterpret_cast : 포인터나 참조를 다른 타입으로 변환할 때 사용. 안전하지 않은 변환이므로 주의가 필요.
//static_cast : 기본적인 타입 변환에 사용. 안전한 변환이지만, 포인터나 참조 간의 변환에는 사용할 수 없음.
//dynamic_cast : 주로 상속 관계에 있는 클래스 간의 변환에 사용. 런타임 타입 체크를 수행하여 안전한 변환을 보장하지만, 성능이 떨어질 수 있음.
//const_cast : const 속성을 제거하거나 추가할 때 사용. 객체의 상수성(constness)을 변경할 때 사용하지만, 잘못 사용하면 프로그램의 안정성을 해칠 수 있음.