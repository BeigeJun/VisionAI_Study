#pragma once
#include <string>
#include <vector>

class IInference {
public:
    virtual ~IInference() = default;

    // =0의 뜻은 가상함수이며, 상속받은 자식 클래스에서 구현해야한다.
    virtual bool bLoad(const std::string& strModelPath, const std::string& strDevice = "CPU", const int nHeight = 900, const int nWidth = 900)=0;
    virtual std::vector<float> vecInfer(const std::vector<float>& vecInputData) = 0;
    virtual std::vector<size_t> vecReturnInputShape() = 0;
    virtual std::vector<size_t> vecReturnOutputShape() = 0;
    virtual int nReturnClassNum() = 0;
};