#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <atlimage.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <algorithm>

namespace fs = std::filesystem;
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "[ERROR] " << msg << std::endl;
    }
} gLogger;

void ApplyAlphaBlend(CImage& img, int x, int y, COLORREF color, float alpha) {
    COLORREF base = img.GetPixel(x, y);
    BYTE r = (BYTE)(GetRValue(color) * alpha + GetRValue(base) * (1.0f - alpha));
    BYTE g = (BYTE)(GetGValue(color) * alpha + GetGValue(base) * (1.0f - alpha));
    BYTE b = (BYTE)(GetBValue(color) * alpha + GetBValue(base) * (1.0f - alpha));
    img.SetPixel(x, y, RGB(r, g, b));
}

int main() {
    const int W = 512, H = 512, C = 3;
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float std[] = { 0.229f, 0.224f, 0.225f };
    const char* inputName = "input";
    const char* outputName = "output";

    std::string enginePath = "D:/0. Model_Save_Folder/Model_Save_Folder_TransUNet/model.trt";
    std::string testDir = "D:/1. DataSet/CppImage";

    std::ifstream file(enginePath, std::ios::binary);
    if (!file) return -1;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    std::unique_ptr<IRuntime> runtime{ createInferRuntime(gLogger) };
    std::unique_ptr<ICudaEngine> engine{ runtime->deserializeCudaEngine(engineData.data(), size) };
    std::unique_ptr<IExecutionContext> context{ engine->createExecutionContext() };

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* dInput, * dOutput;
    size_t tensorBytes = 1 * C * H * W * sizeof(float);
    cudaMalloc(&dInput, tensorBytes);
    cudaMalloc(&dOutput, tensorBytes);

    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png") continue;

        CImage srcImg;
        if (FAILED(srcImg.Load(entry.path().c_str()))) continue;

        CImage img512;
        img512.Create(W, H, 24);
        HDC hdc = img512.GetDC();
        SetStretchBltMode(hdc, HALFTONE);
        srcImg.StretchBlt(hdc, 0, 0, W, H, 0, 0, srcImg.GetWidth(), srcImg.GetHeight(), SRCCOPY);
        img512.ReleaseDC();

        std::vector<float> hInput(C * H * W);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                COLORREF pix = img512.GetPixel(x, y);
                hInput[0 * (H * W) + y * W + x] = ((GetRValue(pix) / 255.0f) - mean[0]) / std[0];
                hInput[1 * (H * W) + y * W + x] = ((GetGValue(pix) / 255.0f) - mean[1]) / std[1];
                hInput[2 * (H * W) + y * W + x] = ((GetBValue(pix) / 255.0f) - mean[2]) / std[2];
            }
        }

        cudaMemcpyAsync(dInput, hInput.data(), tensorBytes, cudaMemcpyHostToDevice, stream);
        context->setInputTensorAddress(inputName, dInput);
        context->setOutputTensorAddress(outputName, dOutput);

        if (!context->enqueueV3(stream)) continue;

        std::vector<float> hOutput(C * H * W);
        cudaMemcpyAsync(hOutput.data(), dOutput, tensorBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        CImage resImg;
        resImg.Create(W, H, 24);
        HDC hdcRes = resImg.GetDC();
        img512.BitBlt(hdcRes, 0, 0);
        resImg.ReleaseDC();

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int bestClass = 0;
                float maxVal = hOutput[0 * (H * W) + y * W + x];
                for (int c = 1; c < C; c++) {
                    float val = hOutput[c * (H * W) + y * W + x];
                    if (val > maxVal) { maxVal = val; bestClass = c; }
                }

                if (bestClass == 1) {
                    ApplyAlphaBlend(resImg, x, y, RGB(0, 255, 0), 0.4f);
                }
                else if (bestClass == 2) {
                    ApplyAlphaBlend(resImg, x, y, RGB(255, 0, 0), 0.4f);
                }
            }
        }

        std::wstring savePath = entry.path().parent_path().wstring() + L"/res_" + entry.path().filename().wstring();
        resImg.Save(savePath.c_str());

        img512.Destroy(); resImg.Destroy(); srcImg.Destroy();
    }

    cudaStreamDestroy(stream);
    cudaFree(dInput); cudaFree(dOutput);
    return 0;
}