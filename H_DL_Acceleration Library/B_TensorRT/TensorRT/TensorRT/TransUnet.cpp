#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <memory>

namespace fs = std::filesystem;

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity eSeverity, const char* pszMsg) noexcept override
    {
        if (eSeverity <= Severity::kWARNING)
            std::cerr << "[TRT] " << pszMsg << std::endl;
    }
};

static const std::vector<std::string> g_vstrClassNames = { "Good", "Broken", "Contamination" };
static const std::vector<cv::Scalar>  g_vscClassColors = {
    cv::Scalar(255,   0,   0),
    cv::Scalar(0, 255,   0),
    cv::Scalar(0,   0, 255)
};
static const float g_afMean[3] = { 0.485f, 0.456f, 0.406f };
static const float g_afStd[3] = { 0.229f, 0.224f, 0.225f };

std::vector<float> Preprocess(const cv::Mat& matBgrImg, int nInputSize)
{
    cv::Mat matResized, matRgb, matRgbF32;
    cv::resize(matBgrImg, matResized, cv::Size(nInputSize, nInputSize), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(matResized, matRgb, cv::COLOR_BGR2RGB);
    matRgb.convertTo(matRgbF32, CV_32FC3, 1.0f / 255.0f);
    matRgbF32 -= cv::Scalar(g_afMean[0], g_afMean[1], g_afMean[2]);
    matRgbF32 /= cv::Scalar(g_afStd[0], g_afStd[1], g_afStd[2]);

    const int nHW = nInputSize * nInputSize;
    std::vector<float> vfBlob(3 * nHW);
    std::vector<cv::Mat> vmatCh(3);
    for (int nC = 0; nC < 3; ++nC)
        vmatCh[nC] = cv::Mat(nInputSize, nInputSize, CV_32FC1, vfBlob.data() + nC * nHW);
    cv::split(matRgbF32, vmatCh);
    return vfBlob;
}

cv::Mat GetArgmax(const float* pfData, int nNumClasses, int nH, int nW)
{
    cv::Mat matPred(nH, nW, CV_8U);
    const int nHW = nH * nW;
    for (int nY = 0; nY < nH; ++nY)
    {
        uchar* pRow = matPred.ptr<uchar>(nY);
        for (int nX = 0; nX < nW; ++nX)
        {
            int   nBestCls = 0;
            float fBestVal = pfData[nY * nW + nX];
            for (int nC = 1; nC < nNumClasses; ++nC)
            {
                float fVal = pfData[nC * nHW + nY * nW + nX];
                if (fVal > fBestVal) { fBestVal = fVal; nBestCls = nC; }
            }
            pRow[nX] = static_cast<uchar>(nBestCls);
        }
    }
    return matPred;
}

cv::Mat DrawResults(const cv::Mat& matBgrImg, const cv::Mat& matPredMask)
{
    CV_Assert(matBgrImg.size() == matPredMask.size());
    cv::Mat matOverlay = matBgrImg.clone();
    cv::Mat matResult = matBgrImg.clone();

    for (int nClsId = 1; nClsId < static_cast<int>(g_vstrClassNames.size()); ++nClsId)
    {
        cv::Mat matClsMask;
        cv::compare(matPredMask, static_cast<uchar>(nClsId), matClsMask, cv::CMP_EQ);
        matOverlay.setTo(g_vscClassColors[nClsId], matClsMask);

        std::vector<std::vector<cv::Point>> vvptContours;
        cv::findContours(matClsMask, vvptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& vptCnt : vvptContours)
        {
            if (cv::contourArea(vptCnt) < 10.0) continue;
            cv::Rect rectBbox = cv::boundingRect(vptCnt);
            cv::rectangle(matResult, rectBbox, g_vscClassColors[nClsId], 1);
            cv::putText(matResult, g_vstrClassNames[nClsId],
                cv::Point(rectBbox.x, max(rectBbox.y - 5, 0)),
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

std::vector<char> LoadEngineFile(const std::string& strPath)
{
    std::ifstream ifs(strPath, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Engine file open failed: " + strPath);
    const size_t nSize = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<char> vcBuffer(nSize);
    ifs.read(vcBuffer.data(), static_cast<std::streamsize>(nSize));
    return vcBuffer;
}

int main()
{
    const std::string strEnginePath = "D:/0. Model_Save_Folder/Model_Save_Folder_HA/model.trt";
    const std::string strDataPath = "D:/1. DataSet/CppImage";
    const int         nInputSize = 900;

    TrtLogger trtLogger;
    std::unique_ptr<nvinfer1::IRuntime> upRuntime(
        nvinfer1::createInferRuntime(trtLogger));
    if (!upRuntime) { std::cerr << "createInferRuntime failed" << std::endl; return EXIT_FAILURE; }

    std::vector<char> vcEngineData = LoadEngineFile(strEnginePath);
    std::unique_ptr<nvinfer1::ICudaEngine> upEngine(
        upRuntime->deserializeCudaEngine(vcEngineData.data(), vcEngineData.size()));
    if (!upEngine) { std::cerr << "deserializeCudaEngine failed" << std::endl; return EXIT_FAILURE; }

    std::unique_ptr<nvinfer1::IExecutionContext> upContext(
        upEngine->createExecutionContext());
    if (!upContext) { std::cerr << "createExecutionContext failed" << std::endl; return EXIT_FAILURE; }

    const int nNbIO = upEngine->getNbIOTensors();
    std::string strInputName, strOutputName;
    for (int i = 0; i < nNbIO; ++i)
    {
        const char* pszName = upEngine->getIOTensorName(i);
        if (upEngine->getTensorIOMode(pszName) == nvinfer1::TensorIOMode::kINPUT)
            strInputName = pszName;
        else
            strOutputName = pszName;
    }
    std::cout << "Input  tensor : " << strInputName << std::endl;
    std::cout << "Output tensor : " << strOutputName << std::endl;

    nvinfer1::Dims4 dimsInput(1, 3, nInputSize, nInputSize);
    upContext->setInputShape(strInputName.c_str(), dimsInput);

    nvinfer1::Dims dimsOutput = upContext->getTensorShape(strOutputName.c_str());
    const int nNumClasses = static_cast<int>(dimsOutput.d[1]);
    const int nOutH = static_cast<int>(dimsOutput.d[2]);
    const int nOutW = static_cast<int>(dimsOutput.d[3]);

    std::cout << "NumClasses=" << nNumClasses
        << "  OutH=" << nOutH << "  OutW=" << nOutW << std::endl;

    const size_t nInputBytes = 1 * 3 * nInputSize * nInputSize * sizeof(float);
    const size_t nOutputBytes = 1 * nNumClasses * nOutH * nOutW * sizeof(float);

    void* pDevInput = nullptr;
    void* pDevOutput = nullptr;
    cudaMalloc(&pDevInput, nInputBytes);
    cudaMalloc(&pDevOutput, nOutputBytes);

    std::vector<float> vfOutputHost(nNumClasses * nOutH * nOutW);

    cudaStream_t cuStream;
    cudaStreamCreate(&cuStream);

    upContext->setTensorAddress(strInputName.c_str(), pDevInput);
    upContext->setTensorAddress(strOutputName.c_str(), pDevOutput);

    std::vector<fs::path> vpathImages = CollectImages(strDataPath);
    if (vpathImages.empty())
    {
        std::cerr << "No images found in: " << strDataPath << std::endl;
        return EXIT_FAILURE;
    }

    LARGE_INTEGER nFreq, nStart, nEnd;
    QueryPerformanceFrequency(&nFreq);

    for (size_t nIdx = 0; nIdx < vpathImages.size(); ++nIdx)
    {
        cv::Mat matImg = cv::imread(vpathImages[nIdx].string());
        if (matImg.empty()) continue;

        std::vector<float> vfBlob = Preprocess(matImg, nInputSize);

        cudaMemcpyAsync(pDevInput, vfBlob.data(), nInputBytes, cudaMemcpyHostToDevice, cuStream);
        cudaStreamSynchronize(cuStream);

        QueryPerformanceCounter(&nStart);

        if (!upContext->enqueueV3(cuStream))
        {
            std::cerr << "enqueueV3 failed on image " << vpathImages[nIdx] << std::endl;
            continue;
        }

        cudaStreamSynchronize(cuStream);
        QueryPerformanceCounter(&nEnd);

        const double dfMs = static_cast<double>(nEnd.QuadPart - nStart.QuadPart)
            * 1000.0 / static_cast<double>(nFreq.QuadPart);

        cudaMemcpyAsync(vfOutputHost.data(), pDevOutput, nOutputBytes, cudaMemcpyDeviceToHost, cuStream);
        cudaStreamSynchronize(cuStream);

        cv::Mat matPredMask = GetArgmax(vfOutputHost.data(), nNumClasses, nOutH, nOutW);

        if (matPredMask.size() != matImg.size())
            cv::resize(matPredMask, matPredMask, matImg.size(), 0, 0, cv::INTER_NEAREST);

        cv::Mat matVis = DrawResults(matImg, matPredMask);

        std::cout << "[" << nIdx + 1 << "/" << vpathImages.size() << "] "
            << vpathImages[nIdx].filename().string()
            << " | Inference: " << dfMs << " ms" << std::endl;

        cv::imshow("Prediction", matVis);
        if (cv::waitKey(0) == 27) break;
    }

    cudaStreamDestroy(cuStream);
    cudaFree(pDevInput);
    cudaFree(pDevOutput);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}