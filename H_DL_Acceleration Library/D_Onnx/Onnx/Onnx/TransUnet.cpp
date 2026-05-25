#include <opencv2/opencv.hpp>
#include "Onnx.h"

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

int main()
{
    try
    {
        const std::string strModelPath = "D:/0. Model_Save_Folder/Model_Save_Folder_HA/model.onnx";
        const std::string strDataPath = "D:/1. DataSet/CppImage";
        const int         nInputSize = 900;

		Onnx onnx(strModelPath, nInputSize, nInputSize);

        std::vector<fs::path> vpathImages = CollectImages(strDataPath);
        if (vpathImages.empty())
        {
            std::cerr << "No images found in: " << strDataPath << std::endl;
            system("pause");
            return EXIT_FAILURE;
        }

        LARGE_INTEGER nFreq, nStart, nEnd;
        QueryPerformanceFrequency(&nFreq);

        for (size_t nIdx = 0; nIdx < vpathImages.size(); ++nIdx)
        {
            QueryPerformanceCounter(&nStart);
            cv::Mat matImg = cv::imread(vpathImages[nIdx].string());
            if (matImg.empty()) continue;

            std::vector<float> vfBlob = Preprocess(matImg, nInputSize);

            std::vector<float> vecOuntPut = onnx.vecInfer(vfBlob);

            QueryPerformanceCounter(&nEnd);
            const double dfMs = (double)(nEnd.QuadPart - nStart.QuadPart) * 1000.0 / (double)nFreq.QuadPart;
			int nClassNum = onnx.nReturnClassNum();

            cv::Mat matPredMask = GetArgmax(vecOuntPut, nClassNum, nInputSize, nInputSize);

            if (matPredMask.size() != matImg.size())
                cv::resize(matPredMask, matPredMask, matImg.size(), 0, 0, cv::INTER_NEAREST);

            cv::Mat matVis = DrawResults(matImg, matPredMask);

            std::cout << "[" << nIdx + 1 << "/" << vpathImages.size() << "] "
                << vpathImages[nIdx].filename().string()
                << " | Inference: " << dfMs << " ms" << std::endl;

            cv::imshow("Prediction", matVis);
            if (cv::waitKey(0) == 27) break;
        }

        cv::destroyAllWindows();
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "[ORT ERROR] " << e.what() << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[STD ERROR] " << e.what() << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}