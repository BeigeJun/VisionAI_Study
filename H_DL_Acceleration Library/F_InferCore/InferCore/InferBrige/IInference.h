#pragma once
#include <string>
#include <vector>

class IInference {
public:
    virtual ~IInference() = default;

    virtual bool Load(const std::string& modelPath) = 0;
    virtual std::vector<float> Predict(const std::vector<float>& input) = 0;
};