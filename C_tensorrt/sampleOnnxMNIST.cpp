#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

// create the network using an ONNX model
class SampleOnnxMNIST
{
	template <typename T>
	using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
		: mParams(params)
		  , mEngine(nullptr)
	{
	}

	bool build();  // function builds the nework engine
	bool infer();

private:
	samplesCommon::OnnxSampleParams mParams;

	nvinfer1::Dims mInputDims;  // The dimensions of input
	nvinfer1::Dims mOutputDims;  // The dimensions of output
	int mNumber{0};

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;  // The TensorRT engine used to run the network
	
	// Parse an ONNX model for MNIST and create a TensorRT network
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, 
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
		SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool processInput(const samplesCommon::BufferManager& buffers); // reads the input and stores the results in a managed buffer

	bool verifyOutput(const samplesCommon::BufferManager& buffers); // Classifies digits and verify results

};


bool SampleOnnxMNIST::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}
	
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}
	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
		builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		return false;
	}

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	assert(mInputDims.nbDims == 4);

	assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	assert(mOutputDims.nbDims == 2);

	return true;

}


// use a onnx parser to create the onnx mnist network and mark the output layer

bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
	SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
	SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(
		locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
		static_cast<int>(gLogger.getReportableSeverity()));

	if (!parsed)
	{
		return false;
	}

	builder->setMaxBatchSize(mParams.batchSize);
	config->setMaxWorkspaceSize(16_MiB);
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
	}
	
	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
	return true;

}


// run the tensorrt inference engine
bool SampleOnnxMNIST::infer()
{
	samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if(!context)
	{
		return false;
	
	}
	assert(mParams.inputTensorNames.size()==1);
	if(!processInput(buffers))
	{
		return false;
	}

	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
	if(!status)
	{
		return false;
	}

	buffers.copyOutputToHost();

	if(!verifyOutput(buffers))
	{
		return false;
	}

	return true;

}

// read the input and store the result in a managed buffer

bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];

	srand(unsigned(time(nullptr)));
	std::vector<uint8_t> fileData(inputH * inputW);

	mNumber = rand() % 10;
	readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

	gLogInfo << "Input: " << std::endl;
	for (int i=0; i<inputH*inputW; i++)
	{
		gLogInfo << (" .:-=+*#%@"[fileData[i]/26]) << (((i + 1) % inputW) ? "" : "\n");
	}
	gLogInfo << std::endl;

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	for(int i=0;i<inputH*inputW;i++)
	{
		hostDataBuffer[i] = 1.0 - float(fileData[i]/255.0);
	}	
	return true;

}


bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
	const int outputSize = mOutputDims.d[1];
	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	float val{0.0f};
	int idx{0};

	float sum{0.0f};
	for (int i=0;i<outputSize;i++)
	{
		output[i]=exp(output[i]);
		sum += output[i];
	}

	gLogInfo << "Output: " << std::endl;
	for (int i=0;i<outputSize;i++)
	{
		output[i] /= sum;
		val = std::max(val, output[i]);
		if (val == output[i])
		{
			idx = i;
		}
	gLogInfo << "Prob "  << i << " " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " " << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
	}
	gLogInfo << std::endl;
	return idx == mNumber && val > 0.9f;
}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
	samplesCommon::OnnxSampleParams params;
	if(args.dataDirs.empty())
	{
		params.dataDirs.push_back("../mnist/");
	}
	else
	{
		params.dataDirs = args.dataDirs;
	}
	params.onnxFileName = "mnist.onnx";
	params.inputTensorNames.push_back("Input3");
	params.batchSize = 1;
	params.outputTensorNames.push_back("Plus214_Output_0");
	params.dlaCore = args.useDLACore;

	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;
	return params;
}

void printHelpInfo()
{
	std::cout
		<< "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
		<< std::endl;

	std::cout << "--help 		Display help informarion" << std::endl;
	std::cout << "--datadir 	Specify path to a data directory, overriding the default. This option can be used "
		"multiple times to add multiple directories. If no data directories are given, the default is to use" 
		"(data/samples/mnist/, /data/mnist/)"
		<< std::endl;
}


int main(int argc, char** argv)
{
	samplesCommon::Args args;
	bool argsOK = samplesCommon::parseArgs(args, argc, argv);
	if(!argsOK)
	{
		gLogError << "Invalid arguments " << std::endl;
		printHelpInfo();
		return EXIT_FAILURE;
	}
	if(args.help)
	{
		printHelpInfo();
		return EXIT_SUCCESS;
	}
	auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
	gLogger.reportTestStart(sampleTest);

	SampleOnnxMNIST sample(initializeSampleParams(args));
	gLogInfo << "Building and running a GPU inference engine for onnx MNIST" << std::endl;

	if (!sample.build())
	{
		return gLogger.reportFail(sampleTest);
	}
	if (!sample.infer())
	{
		return gLogger.reportFail(sampleTest);
	}
	return gLogger.reportPass(sampleTest);


}


