#define NOMINMAX

#include <windows.h>
#include <atlimage.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>

namespace fs = std::filesystem;

static constexpr int nNumClasses = 3;
static constexpr int nInputH = 900;
static constexpr int nInputW = 900;
static constexpr float fOverlayAlpha = 0.4f;

static const BYTE pClsB[nNumClasses] = { 255, 0, 0 };
static const BYTE pClsG[nNumClasses] = { 0, 255, 0 };
static const BYTE pClsR[nNumClasses] = { 0, 0, 255 };

static const wchar_t* pSzClsName[nNumClasses] =
{
    L"Good",
    L"Broken",
    L"Contamination"
};

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* pSzMsg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::wcerr << L"[TRT] " << pSzMsg << L"\n";
    }
};

static TrtLogger stLogger;

struct BgrImageInfo
{
    std::vector<unsigned char> vecBuffer;
    unsigned int nWidth = 0;
    unsigned int nHeight = 0;
    unsigned int nWidthStep = 0;
    unsigned int nChannel = 0;
};

static bool bLoadFileToBgr(const wchar_t* pSzFileName, int nNeedW, int nNeedH, BgrImageInfo& stOutInfo)
{
    CImage stImage;
    HRESULT hResult = stImage.Load(pSzFileName);
    if (FAILED(hResult)) return false;

    if (stImage.GetBPP() != 24)
    {
        CImage stConv;
        stConv.Create(stImage.GetWidth(), stImage.GetHeight(), 24);
        stImage.BitBlt(stConv.GetDC(), 0, 0);
        stConv.ReleaseDC();
        stImage.Destroy();
        stImage.Attach(stConv.Detach());
    }

    if (stImage.GetWidth() != nNeedW || stImage.GetHeight() != nNeedH)
    {
        CImage stResized;
        stResized.Create(nNeedW, nNeedH, 24);
        HDC hDC = stResized.GetDC();
        SetStretchBltMode(hDC, HALFTONE);
        stImage.StretchBlt(hDC, 0, 0, nNeedW, nNeedH, SRCCOPY);
        stResized.ReleaseDC();
        stImage.Destroy();
        stImage.Attach(stResized.Detach());
    }

    stOutInfo.nWidth = (unsigned int)stImage.GetWidth();
    stOutInfo.nHeight = (unsigned int)stImage.GetHeight();
    stOutInfo.nChannel = 3;
    stOutInfo.nWidthStep = stOutInfo.nWidth * stOutInfo.nChannel;
    stOutInfo.vecBuffer.resize((size_t)stOutInfo.nWidthStep * stOutInfo.nHeight);

    for (unsigned int nY = 0; nY < stOutInfo.nHeight; ++nY)
    {
        BYTE* pRow = (BYTE*)stImage.GetPixelAddress(0, (int)nY);
        if (!pRow) return false;
        std::memcpy(stOutInfo.vecBuffer.data() + (size_t)nY * stOutInfo.nWidthStep, pRow, stOutInfo.nWidthStep);
    }
    return true;
}

static void vBgrImageToNhwcFloatTensor(const BgrImageInfo& stIm, std::vector<float>& vecInputTensor, bool bUseNormalize = true)
{
    const float pMean[3] = { 0.485f, 0.456f, 0.406f };
    const float pStdDev[3] = { 0.229f, 0.224f, 0.225f };
    const float fNormalizeValue = 1.0f / 255.0f;

    const int64_t nWidth = stIm.nWidth;
    const int64_t nHeight = stIm.nHeight;
    const int64_t nChannel = stIm.nChannel;

    vecInputTensor.resize((size_t)nWidth * nHeight * nChannel);

    for (int64_t nY = 0; nY < nHeight; ++nY)
    {
        const unsigned char* pSrc = stIm.vecBuffer.data() + (size_t)nY * stIm.nWidthStep;
        float* pDst = vecInputTensor.data() + (size_t)nY * nWidth * nChannel;
        for (int64_t nX = 0; nX < nWidth; ++nX)
        {
            float fR = (float)(pSrc[nX * 3 + 2]) * fNormalizeValue;
            float fG = (float)(pSrc[nX * 3 + 1]) * fNormalizeValue;
            float fB = (float)(pSrc[nX * 3 + 0]) * fNormalizeValue;

            if (bUseNormalize) {
                pDst[nX * 3 + 0] = (fR - pMean[0]) / pStdDev[0];
                pDst[nX * 3 + 1] = (fG - pMean[1]) / pStdDev[1];
                pDst[nX * 3 + 2] = (fB - pMean[2]) / pStdDev[2];
            }
            else {
                pDst[nX * 3 + 0] = fR; pDst[nX * 3 + 1] = fG; pDst[nX * 3 + 2] = fB;
            }
        }
    }
}

static void vNhwcToNchw(const std::vector<float>& vecNhwc, std::vector<float>& vecNchw, int nH, int nW)
{
    vecNchw.resize((size_t)3 * nH * nW);
    const size_t nPlaneSize = (size_t)nH * nW;
    float* pCh0 = vecNchw.data();
    float* pCh1 = vecNchw.data() + nPlaneSize;
    float* pCh2 = vecNchw.data() + nPlaneSize * 2;

    for (int nI = 0; nI < (int)nPlaneSize; ++nI) {
        pCh0[nI] = vecNhwc[nI * 3 + 0];
        pCh1[nI] = vecNhwc[nI * 3 + 1];
        pCh2[nI] = vecNhwc[nI * 3 + 2];
    }
}

static void vArgmaxMask(const float* pOutput, int nH, int nW, std::vector<BYTE>& vecMask)
{
    const size_t nPlaneSize = (size_t)nH * nW;
    vecMask.resize(nPlaneSize);
    for (size_t nI = 0; nI < nPlaneSize; ++nI) {
        int nBestCls = 0;
        float fMaxVal = pOutput[nI];
        for (int nC = 1; nC < nNumClasses; ++nC) {
            const float fVal = pOutput[nC * nPlaneSize + nI];
            if (fVal > fMaxVal) { fMaxVal = fVal; nBestCls = nC; }
        }
        vecMask[nI] = (BYTE)nBestCls;
    }
}

struct BlobRect {
    int minX, minY, maxX, maxY;
};

static void vDrawResults(const BgrImageInfo& stSrcInfo, const BYTE* pMask, CImage& stResultImage)
{
    const int nH = (int)stSrcInfo.nHeight;
    const int nW = (int)stSrcInfo.nWidth;

    stResultImage.Destroy();
    stResultImage.Create(nW, nH, 24);

    int nPitch = stResultImage.GetPitch();
    BYTE* pDstBits = (BYTE*)stResultImage.GetBits();

    for (int nY = 0; nY < nH; ++nY) {
        BYTE* pDstRow = pDstBits + (nY * nPitch);
        const BYTE* pSrcRow = stSrcInfo.vecBuffer.data() + (size_t)nY * stSrcInfo.nWidthStep;

        for (int nX = 0; nX < nW; ++nX) {
            const BYTE nCls = pMask[(size_t)nY * nW + nX];
            if (nCls == 0) {
                pDstRow[nX * 3 + 0] = pSrcRow[nX * 3 + 0];
                pDstRow[nX * 3 + 1] = pSrcRow[nX * 3 + 1];
                pDstRow[nX * 3 + 2] = pSrcRow[nX * 3 + 2];
            }
            else {
                pDstRow[nX * 3 + 0] = (BYTE)(pClsB[nCls] * fOverlayAlpha + pSrcRow[nX * 3 + 0] * (1.f - fOverlayAlpha));
                pDstRow[nX * 3 + 1] = (BYTE)(pClsG[nCls] * fOverlayAlpha + pSrcRow[nX * 3 + 1] * (1.f - fOverlayAlpha));
                pDstRow[nX * 3 + 2] = (BYTE)(pClsR[nCls] * fOverlayAlpha + pSrcRow[nX * 3 + 2] * (1.f - fOverlayAlpha));
            }
        }
    }

    HDC hDC = stResultImage.GetDC();
    HFONT hFont = CreateFontW(14, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Arial");
    HFONT hOldFont = (HFONT)SelectObject(hDC, hFont);
    SetBkMode(hDC, TRANSPARENT);

    std::vector<bool> vecVisited(nH * nW, false);

    for (int nY = 0; nY < nH; ++nY) {
        for (int nX = 0; nX < nW; ++nX) {
            size_t nIdx = (size_t)nY * nW + nX;
            BYTE nCls = pMask[nIdx];

            if (nCls > 0 && !vecVisited[nIdx]) {
                BlobRect rect = { nX, nY, nX, nY };
                std::queue<std::pair<int, int>> q;

                q.push({ nX, nY });
                vecVisited[nIdx] = true;

                int nPixelCount = 0;
                while (!q.empty()) {
                    std::pair<int, int> curr = q.front();
                    q.pop();
                    nPixelCount++;

                    if (curr.first < rect.minX) rect.minX = curr.first;
                    if (curr.first > rect.maxX) rect.maxX = curr.first;
                    if (curr.second < rect.minY) rect.minY = curr.second;
                    if (curr.second > rect.maxY) rect.maxY = curr.second;

                    int dx[] = { 0, 0, 1, -1 };
                    int dy[] = { 1, -1, 0, 0 };

                    for (int i = 0; i < 4; ++i) {
                        int nextX = curr.first + dx[i];
                        int nextY = curr.second + dy[i];

                        if (nextX >= 0 && nextX < nW && nextY >= 0 && nextY < nH) {
                            size_t nextIdx = (size_t)nextY * nW + nextX;
                            if (pMask[nextIdx] == nCls && !vecVisited[nextIdx]) {
                                vecVisited[nextIdx] = true;
                                q.push({ nextX, nextY });
                            }
                        }
                    }
                }

                if (nPixelCount < 10) continue;

                COLORREF color = RGB(pClsR[nCls], pClsG[nCls], pClsB[nCls]);
                HPEN hPen = CreatePen(PS_SOLID, 2, color);
                HPEN hOldPen = (HPEN)SelectObject(hDC, hPen);
                SelectObject(hDC, GetStockObject(NULL_BRUSH));

                Rectangle(hDC, rect.minX, rect.minY, rect.maxX + 1, rect.maxY + 1);

                SetTextColor(hDC, color);
                TextOutW(hDC, rect.minX + 2, (rect.minY > 18 ? rect.minY - 18 : rect.minY + 2),
                    pSzClsName[nCls], (int)wcslen(pSzClsName[nCls]));

                SelectObject(hDC, hOldPen);
                DeleteObject(hPen);
            }
        }
    }

    SelectObject(hDC, hOldFont);
    DeleteObject(hFont);
    stResultImage.ReleaseDC();
}

static nvinfer1::ICudaEngine* pLoadEngine(const wchar_t* pSzTrtPath, nvinfer1::IRuntime*& pRuntime)
{
    char pSzPathA[MAX_PATH] = {};
    WideCharToMultiByte(CP_ACP, 0, pSzTrtPath, -1, pSzPathA, MAX_PATH, nullptr, nullptr);
    std::ifstream stFile(pSzPathA, std::ios::binary | std::ios::ate);
    
    if (!stFile.is_open()) 
        return nullptr;
    
    const size_t nFileSize = (size_t)stFile.tellg();

    stFile.seekg(0);
    std::vector<char> vecEngineData(nFileSize);
    stFile.read(vecEngineData.data(), nFileSize);
    pRuntime = nvinfer1::createInferRuntime(stLogger);
    return pRuntime->deserializeCudaEngine(vecEngineData.data(), nFileSize);
}

static std::vector<fs::path> vecCollectImages(const fs::path& stDirPath)
{
    std::vector<fs::path> vecImageList;
    if (!fs::exists(stDirPath))
        return vecImageList;

    for (const auto& stEntry : fs::directory_iterator(stDirPath)) {
        if (!stEntry.is_regular_file()) continue;
        auto wExt = stEntry.path().extension().wstring();
        std::transform(wExt.begin(), wExt.end(), wExt.begin(), ::towlower);
        if (wExt == L".png" || wExt == L".jpg" || wExt == L".jpeg" || wExt == L".bmp")
            vecImageList.push_back(stEntry.path());
    }

    std::sort(vecImageList.begin(), vecImageList.end());
    return vecImageList;
}

int wmain(int argc, wchar_t* argv[])
{
    const wchar_t* pSzTrtPath = L"D:/0. Model_Save_Folder/Model_Save_Folder_Ha/model.trt";
    const fs::path stInputDir = L"D:/1. DataSet/CppImage";
    const fs::path stOutputDir = L"D:/1. DataSet/CppImage";

    LARGE_INTEGER stPerfFreq;
    QueryPerformanceFrequency(&stPerfFreq);
    fs::create_directories(stOutputDir);

    nvinfer1::IRuntime* pRuntime = nullptr;
    nvinfer1::ICudaEngine* pEngine = pLoadEngine(pSzTrtPath, pRuntime);
    if (!pEngine) return 1;

    nvinfer1::IExecutionContext* pContext = pEngine->createExecutionContext();
    const char* pSzInName = pEngine->getIOTensorName(0);
    const char* pSzOutName = pEngine->getIOTensorName(1);

    const size_t nInputBytes = sizeof(float) * 1 * 3 * nInputH * nInputW;
    const size_t nOutputBytes = sizeof(float) * 1 * nNumClasses * nInputH * nInputW;

    void* dInput = nullptr;
    void* dOutput = nullptr;
    cudaMalloc(&dInput, nInputBytes);
    cudaMalloc(&dOutput, nOutputBytes);

    cudaStream_t stCudaStream;
    cudaStreamCreate(&stCudaStream);

    pContext->setTensorAddress(pSzInName, dInput);
    pContext->setTensorAddress(pSzOutName, dOutput);

    std::vector<float> hOutput(nNumClasses * nInputH * nInputW);
    auto vecImagePaths = vecCollectImages(stInputDir);
    if (vecImagePaths.empty()) return 0;

    std::wcout << L"Total: " << vecImagePaths.size() << L"\n";

    for (size_t nI = 0; nI < vecImagePaths.size(); ++nI)
    {
        const fs::path& stCurrentPath = vecImagePaths[nI];
        std::wcout << L"[" << (nI + 1) << L"/" << vecImagePaths.size() << L"] Process: " << stCurrentPath.filename().wstring() << L"\n";

        BgrImageInfo stBgrInfo;
        if (!bLoadFileToBgr(stCurrentPath.wstring().c_str(), nInputW, nInputH, stBgrInfo)) continue;

        std::vector<float> vecNhwc, vecNchw;
        vBgrImageToNhwcFloatTensor(stBgrInfo, vecNhwc, true);
        vNhwcToNchw(vecNhwc, vecNchw, nInputH, nInputW);

        LARGE_INTEGER stStart, stEnd;
        QueryPerformanceCounter(&stStart);

        cudaMemcpyAsync(dInput, vecNchw.data(), nInputBytes, cudaMemcpyHostToDevice, stCudaStream);
        pContext->enqueueV3(stCudaStream);
        cudaMemcpyAsync(hOutput.data(), dOutput, nOutputBytes, cudaMemcpyDeviceToHost, stCudaStream);

        cudaStreamSynchronize(stCudaStream);
        QueryPerformanceCounter(&stEnd);

        double fElapsedMs = (double)(stEnd.QuadPart - stStart.QuadPart) * 1000.0 / stPerfFreq.QuadPart;
        std::wcout << L"  Infer: " << fElapsedMs << L" ms\n";

        std::vector<BYTE> vecResultMask;
        vArgmaxMask(hOutput.data(), nInputH, nInputW, vecResultMask);

        CImage stFinalImage;
        vDrawResults(stBgrInfo, vecResultMask.data(), stFinalImage);

        fs::path stSavePath = stOutputDir / (L"pred_" + stCurrentPath.stem().wstring() + L".png");
        stFinalImage.Save(stSavePath.wstring().c_str());
    }

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaStreamDestroy(stCudaStream);
    delete pContext;
    delete pEngine;
    delete pRuntime;

    return 0;
}