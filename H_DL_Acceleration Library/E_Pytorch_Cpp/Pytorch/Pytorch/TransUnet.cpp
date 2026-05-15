#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <Windows.h>

namespace fs = std::filesystem;

static const std::vector<std::string> g_vstrClassNames = { "Good", "Broken", "Contamination" };
static const std::vector<cv::Scalar> g_vscClassColors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255) };
static const float g_afMean[3] = { 0.485f, 0.456f, 0.406f };
static const float g_afStd[3] = { 0.229f, 0.224f, 0.225f };

torch::Tensor Preprocess(const cv::Mat& matBgrImg, int nInputH, int nInputW)
{
    cv::Mat matResized, matRgb, matRgbF32;

    cv::resize(matBgrImg, matResized, cv::Size(nInputW, nInputH), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(matResized, matRgb, cv::COLOR_BGR2RGB);
    matRgb.convertTo(matRgbF32, CV_32FC3, 1.0f / 255.0f);

    auto tImg = torch::from_blob(matRgbF32.data, { nInputH, nInputW, 3 }, torch::kFloat32).clone();
    tImg = tImg.permute({ 2, 0, 1 });

    for (int nC = 0; nC < 3; ++nC)
    {
        tImg[nC] = tImg[nC].sub(g_afMean[nC]).div(g_afStd[nC]);
    }

    return tImg.unsqueeze(0);
}

cv::Mat GetArgmax(const torch::Tensor& tOutput)
{
    auto tPred = tOutput.squeeze(0).argmax(0).to(torch::kUInt8).contiguous();

    int nH = (int)tPred.size(0);
    int nW = (int)tPred.size(1);

    cv::Mat matMask(nH, nW, CV_8U);
    std::memcpy(matMask.data, tPred.data_ptr<uint8_t>(), (size_t)nH * nW * sizeof(uint8_t));
    return matMask;
}

cv::Mat DrawResults(const cv::Mat& matBgrImg, const cv::Mat& matPredMask)
{
    CV_Assert(matBgrImg.size() == matPredMask.size());

    cv::Mat matOverlay = matBgrImg.clone();
    cv::Mat matResult = matBgrImg.clone();

    for (int nClsId = 1; nClsId < (int)g_vstrClassNames.size(); ++nClsId)
    {
        cv::Mat matClsMask;
        cv::compare(matPredMask, (uchar)nClsId, matClsMask, cv::CMP_EQ);

        matOverlay.setTo(g_vscClassColors[nClsId], matClsMask);

        std::vector<std::vector<cv::Point>> vvptContours;
        cv::findContours(matClsMask, vvptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& vptCnt : vvptContours)
        {
            if (cv::contourArea(vptCnt) < 10.0) continue;

            cv::Rect rectBbox = cv::boundingRect(vptCnt);
            cv::rectangle(matResult, rectBbox, g_vscClassColors[nClsId], 1);
            cv::putText(matResult, g_vstrClassNames[nClsId], cv::Point(rectBbox.x, max(rectBbox.y - 5, 0)),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, g_vscClassColors[nClsId], 1, cv::LINE_AA);
        }
    }

    cv::Mat matBlended;
    cv::addWeighted(matOverlay, 0.4, matResult, 0.6, 0.0, matBlended);
    return matBlended;
}

std::vector<fs::path> CollectImages(const std::string& strDataPath)
{
    std::vector<fs::path> vpathImages;
    if (!fs::exists(strDataPath)) return vpathImages;

    for (const auto& entry : fs::recursive_directory_iterator(strDataPath))
    {
        if (!entry.is_regular_file()) continue;
        std::string strExt = entry.path().extension().string();
        std::transform(strExt.begin(), strExt.end(), strExt.begin(), ::tolower);
        if (strExt == ".png" || strExt == ".jpg" || strExt == ".jpeg" || strExt == ".bmp")
            vpathImages.push_back(entry.path());
    }
    std::sort(vpathImages.begin(), vpathImages.end());
    return vpathImages;
}

int main()
{
    const std::string strModelPath = "D:/0. Model_Save_Folder/Model_Save_Folder_HA/model_cuda.pt";
    const std::string strDataPath = "D:/1. DataSet/CppImage";
    const int nInputSize = 900;

    const torch::DeviceType eDeviceType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device device(eDeviceType);

    std::cout << "Device: " << (eDeviceType == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(strModelPath, device);
        module.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "Load Fail: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<fs::path> vpathImages = CollectImages(strDataPath);
    if (vpathImages.empty()) return EXIT_FAILURE;

    LARGE_INTEGER nFreq, nStart, nEnd;
    QueryPerformanceFrequency(&nFreq);

    for (size_t i = 0; i < vpathImages.size(); ++i)
    {
        cv::Mat matImg = cv::imread(vpathImages[i].string());
        if (matImg.empty()) continue;

        torch::Tensor tInput = Preprocess(matImg, nInputSize, nInputSize);
        tInput = tInput.to(device);

        if (eDeviceType == torch::kCUDA) torch::cuda::synchronize();

        QueryPerformanceCounter(&nStart);
        torch::Tensor tOutput;
        {
            torch::NoGradGuard noGrad;
            tOutput = module.forward({ tInput }).toTensor();
        }
        if (eDeviceType == torch::kCUDA) torch::cuda::synchronize();
        QueryPerformanceCounter(&nEnd);

        double dfMs = (double)(nEnd.QuadPart - nStart.QuadPart) * 1000.0 / (double)nFreq.QuadPart;

        tOutput = tOutput.to(torch::kCPU);
        cv::Mat matPredMask = GetArgmax(tOutput);

        if (matPredMask.size() != matImg.size())
            cv::resize(matPredMask, matPredMask, matImg.size(), 0, 0, cv::INTER_NEAREST);

        cv::Mat matVis = DrawResults(matImg, matPredMask);

        std::cout << "[" << i + 1 << "/" << vpathImages.size() << "] "
            << vpathImages[i].filename().string() << " | Inference: " << dfMs << " ms" << std::endl;

        cv::imshow("Prediction", matVis);
        if (cv::waitKey(0) == 27) break;
    }

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}