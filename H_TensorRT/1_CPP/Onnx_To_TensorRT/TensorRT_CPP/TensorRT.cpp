#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

// TensorRT 변환 과정에서 발생한 오류 출력
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main() {
    std::string strBasePath = "D:/0. Model_Save_Folder/Model_Save_Folder_Trans";
    std::string strOnnxPath = strBasePath + "/model.onnx";
    std::string strTrtPath = strBasePath + "/model.trt";

    const char* cOnnxFile = strOnnxPath.c_str();

	//Builder 생성 -> GPU 및 하드웨어 확인 밑 커널 최적화 수행. 이 때문에 PC 환경이 다른 경우 빌드된 엔진이 호환되지 않을 수 있음
    IBuilder* pBuilder = createInferBuilder(gLogger);
    //시프트 연산자로 플래그 설정
    uint32_t uFlag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //모델 생성을 위한  네트워크 객체 생성
    INetworkDefinition* pNetwork = pBuilder->createNetworkV2(uFlag);

    //Onnx->TensorRT를 수행할 Paser 객체 생성
    IParser* pParser = createParser(*pNetwork, gLogger);

    //Onnx 가중치를 읽어 pNetwork에 적용
    if (!pParser->parseFromFile(cOnnxFile, static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "ONNX Parsing Fail: " << cOnnxFile << std::endl;
        return -1;
    }

    //TensorRT 설정을 저장할 Config 생성
    IBuilderConfig* pConfig = pBuilder->createBuilderConfig();

    ////FP16 지원 확인 후 가능하다면 적용
    //if (pBuilder->platformHasFastFp16())
    //{
    //    pConfig->setFlag(BuilderFlag::kFP16);
    //    std::cout << "Apply FP16 Optimization" << std::endl;
    //}

    //직렬화된 모델을 메모리에 생성
    std::cout << "Build Start" << std::endl;
    IHostMemory* pSerializedModel = pBuilder->buildSerializedNetwork(*pNetwork, *pConfig);

    if (pSerializedModel == nullptr) {
        std::cerr << "Engine Build Fail" << std::endl;
        return -1;
    }

    std::ofstream osEngineFile(strTrtPath, std::ios::binary);
    if (!osEngineFile) {
        std::cerr << "File Create Fail: " << strTrtPath << std::endl;
        return -1;
    }

	//직렬화된 모델을 파일로 저장
    osEngineFile.write(reinterpret_cast<const char*>(pSerializedModel->data()), pSerializedModel->size());

    //자원 해제
    delete pSerializedModel;
    delete pParser;
    delete pNetwork;
    delete pConfig;
    delete pBuilder;

    std::cout << "Finish" << std::endl;

    return 0;
}