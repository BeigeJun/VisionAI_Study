#include "pch.h"
#include <iostream>
#include "Total_DLL.h"
#include "IInference.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static const std::vector<std::string> g_vstrClassNames = { "Good", "Broken", "Contamination" };
static const std::vector<cv::Scalar> g_vscClassColors = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255)
};

static const float g_fMean[3] = { 0.485f, 0.456f, 0.406f };
static const float g_fStd[3] = { 0.229f, 0.224f, 0.225f };

std::vector<float> Preprocess(const cv::Mat& matBgrImg, int nInputSize) {
    cv::Mat matResized, matRgb, matRgbF32;
    cv::resize(matBgrImg, matResized, cv::Size(nInputSize, nInputSize), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(matResized, matRgb, cv::COLOR_BGR2RGB);
    matRgb.convertTo(matRgbF32, CV_32FC3, 1.0f / 255.0f);

    matRgbF32 -= cv::Scalar(g_fMean[0], g_fMean[1], g_fMean[2]);
    matRgbF32 /= cv::Scalar(g_fStd[0], g_fStd[1], g_fStd[2]);

    int nHW = nInputSize * nInputSize;
    std::vector<float> vfBlob(3 * nHW);
    std::vector<cv::Mat> vmatCh(3);
    for (int i = 0; i < 3; ++i) {
        vmatCh[i] = cv::Mat(nInputSize, nInputSize, CV_32FC1, vfBlob.data() + i * nHW);
    }
    cv::split(matRgbF32, vmatCh);
    return vfBlob;
}

cv::Mat GetArgmax(const std::vector<float>& vecData, int nNumClasses, int nH, int nW) {
    cv::Mat matPred(nH, nW, CV_8U);
    int nHW = nH * nW;
    for (int y = 0; y < nH; ++y) {
        uchar* pRow = matPred.ptr<uchar>(y);
        for (int x = 0; x < nW; ++x) {
            int nBestCls = 0;
            float fBestVal = vecData[y * nW + x];
            for (int c = 1; c < nNumClasses; ++c) {
                float fVal = vecData[c * nHW + y * nW + x];
                if (fVal > fBestVal) { fBestVal = fVal; nBestCls = c; }
            }
            pRow[x] = static_cast<uchar>(nBestCls);
        }
    }
    return matPred;
}

cv::Mat DrawResults(const cv::Mat& matBgrImg, const cv::Mat& matPredMask) {
    cv::Mat matOverlay = matBgrImg.clone();
    cv::Mat matResult = matBgrImg.clone();
    for (int nClsId = 1; nClsId < (int)g_vstrClassNames.size(); ++nClsId) {
        cv::Mat matClsMask;
        cv::compare(matPredMask, nClsId, matClsMask, cv::CMP_EQ);
        matOverlay.setTo(g_vscClassColors[nClsId], matClsMask);
        std::vector<std::vector<cv::Point>> vvptContours;
        cv::findContours(matClsMask, vvptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& vptCnt : vvptContours) {
            if (cv::contourArea(vptCnt) < 10.0) continue;
            cv::Rect rectBbox = cv::boundingRect(vptCnt);
            cv::rectangle(matResult, rectBbox, g_vscClassColors[nClsId], 1);
            cv::putText(matResult, g_vstrClassNames[nClsId], cv::Point(rectBbox.x, std::max(rectBbox.y - 5, 0)),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, g_vscClassColors[nClsId], 1, cv::LINE_AA);
        }
    }
    cv::Mat matBlended;
    cv::addWeighted(matOverlay, 0.4, matResult, 0.6, 0.0, matBlended);
    return matBlended;
}

std::vector<fs::path> CollectImages(const std::string& strDataPath) {
    std::vector<fs::path> vpathPaths;
    if (!fs::exists(strDataPath)) return vpathPaths;
    for (const auto& entry : fs::recursive_directory_iterator(strDataPath)) {
        if (!entry.is_regular_file()) continue;
        std::string strExt = entry.path().extension().string();
        std::transform(strExt.begin(), strExt.end(), strExt.begin(), ::tolower);
        if (strExt == ".png" || strExt == ".jpg" || strExt == ".jpeg" || strExt == ".bmp")
            vpathPaths.push_back(entry.path());
    }
    std::sort(vpathPaths.begin(), vpathPaths.end());
    return vpathPaths;
}

int main() {
    const std::string strModelPath = "D:/0. Model_Save_Folder/Model_Save_Folder_HA/Vino/model.xml";
    const std::string strDataPath = "D:/1. DataSet/CppImage";
    const int nInputSize = 900;
    const std::string strDevice = "GPU";

    try {
        std::unique_ptr<IInference> pInference = Total_DLL::Create(BackendType::OpenVINO);

        if (!pInference->bLoad(strModelPath, strDevice, nInputSize, nInputSize)) {
            std::cerr << "Failed to load model." << std::endl;
            return -1;
        }

        auto vpathImagePaths = CollectImages(strDataPath);

        for (size_t i = 0; i < vpathImagePaths.size(); ++i) {
            cv::Mat matImg = cv::imread(vpathImagePaths[i].string());
            if (matImg.empty()) continue;
            auto vfBlob = Preprocess(matImg, nInputSize);

            std::vector<float> vecResult = pInference->vecInfer(vfBlob);
            int nClassNum = pInference->nReturnClassNum();
            std::vector<size_t> vecOutputShape = pInference->vecReturnOutputShape();

            cv::Mat matPredMask = GetArgmax(vecResult, nClassNum, (int)vecOutputShape[2], (int)vecOutputShape[3]);
            if (matPredMask.size() != matImg.size())
                cv::resize(matPredMask, matPredMask, matImg.size(), 0, 0, cv::INTER_NEAREST);

            cv::Mat matVis = DrawResults(matImg, matPredMask);
            cv::imshow("Prediction", matVis);
            if (cv::waitKey(1) == 27) break;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}