#include "pch.h"
#include "TensorRT.h"

TensorRT::TensorRT(std::string strEnginePath, std::string strDataPath, int nHeight, int nWidth)
{
    m_upRuntime.reset(nvinfer1::createInferRuntime(m_Logger));

    std::ifstream ifs(strEnginePath, std::ios::binary | std::ios::ate);
    const size_t nSize = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<char> vcBuffer(nSize);
    ifs.read(vcBuffer.data(), static_cast<std::streamsize>(nSize));

    m_upEngine.reset(m_upRuntime->deserializeCudaEngine(vcBuffer.data(), vcBuffer.size()));
    m_upContext.reset(m_upEngine->createExecutionContext());

    const int nNbIO = m_upEngine->getNbIOTensors();
    std::string strInputName, strOutputName;
    for (int i = 0; i < nNbIO; ++i)
    {
        const char* pszName = m_upEngine->getIOTensorName(i);
        if (m_upEngine->getTensorIOMode(pszName) == nvinfer1::TensorIOMode::kINPUT)
            strInputName = pszName;
        else
            strOutputName = pszName;
    }

    nvinfer1::Dims4 dimsInput(1, 3, nHeight, nWidth);
    m_upContext->setInputShape(strInputName.c_str(), dimsInput);

    nvinfer1::Dims dimsOutput = m_upContext->getTensorShape(strOutputName.c_str());
    m_nNumClasses = static_cast<int>(dimsOutput.d[1]);
    const int nOutH = static_cast<int>(dimsOutput.d[2]);
    const int nOutW = static_cast<int>(dimsOutput.d[3]);

    m_nInputBytes = 1 * 3 * nHeight * nWidth * sizeof(float);
    m_nOutputBytes = 1 * m_nNumClasses * nOutH * nOutW * sizeof(float);


    cudaMalloc(&m_pDevInput, m_nInputBytes);
    cudaMalloc(&m_pDevOutput, m_nOutputBytes);

    m_vecfOutputHost.resize(m_nNumClasses * nOutH * nOutW);

    cudaStreamCreate(&m_CudaStream);

    m_upContext->setTensorAddress(strInputName.c_str(), m_pDevInput);
    m_upContext->setTensorAddress(strOutputName.c_str(), m_pDevOutput);
}

TensorRT::~TensorRT()
{
    if (m_pDevInput)  cudaFree(m_pDevInput);
    if (m_pDevOutput) cudaFree(m_pDevOutput);
    cudaStreamDestroy(m_CudaStream);
}

std::vector<float> TensorRT::vecInfer(const std::vector<float>& vecInputData)
{
    cudaMemcpyAsync(m_pDevInput, vecInputData.data(), m_nInputBytes, cudaMemcpyHostToDevice, m_CudaStream);
    cudaStreamSynchronize(m_CudaStream);

    m_upContext->enqueueV3(m_CudaStream);

    cudaStreamSynchronize(m_CudaStream);

    cudaMemcpyAsync(m_vecfOutputHost.data(), m_pDevOutput, m_nOutputBytes, cudaMemcpyDeviceToHost, m_CudaStream);
    cudaStreamSynchronize(m_CudaStream);

    return m_vecfOutputHost;
}

void TensorRT::Terminate()
{
    if (m_pDevInput)  cudaFree(m_pDevInput);
    if (m_pDevOutput) cudaFree(m_pDevOutput);
    cudaStreamDestroy(m_CudaStream);
}

int TensorRT::nReturnClassNum() const
{
    return m_nNumClasses;
}