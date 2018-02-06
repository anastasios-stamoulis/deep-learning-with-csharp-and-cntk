// CPPUtil.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "CPPUtil.h"

double version() { return 1.0; }


typedef std::unordered_map<CNTK::Variable, CNTK::ValuePtr> dict_t;

void load_bin(const std::wstring& filepath, std::vector<float>& buffer) {
	std::ifstream fin(filepath, std::ios::in | std::ios::binary);
	fin.read((char*)(&buffer[0]), buffer.size() * sizeof(float));
}

void save_bin(const std::wstring& filepath, std::vector<float>& buffer) {
	std::ofstream fout(filepath, std::ios::out | std::ios::binary);
	if (!fout) {
		std::wcout << "Cannot write into " << filepath << std::endl;
		return;
	}
	fout.write((char*)(&buffer[0]), buffer.size() * sizeof(float));
}

cv::Mat load_image_and_subtract_mean(const std::wstring& path, const cv::Size& image_size, bool subtract_mean) {
#if 1
	std::ifstream stream(path, std::ios::in | std::ios::binary);
	std::vector<uint8_t> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
	cv::Mat image(cv::Size(0, 0), CV_32FC3);
	cv::imdecode(contents, cv::IMREAD_COLOR, &image);
#else
	auto image = cv::imread(path);
#endif
	// resize
	cv::resize(image, image, image_size);
	image.convertTo(image, CV_32FC3);

	// subtract the mean
	if (subtract_mean) {
		//cv::Scalar mean_BGR(103.939, 116.779, 123.68);  // FIXME
		cv::Scalar mean_BGR(110, 110, 110);  // FIXME
		image -= mean_BGR;
	}

	return image;
}

void load_image_in_first_channel_format(const std::wstring& imagePath, const cv::Size& image_size, float* dst, bool subtract_mean) {
	auto image = load_image_and_subtract_mean(imagePath, image_size, subtract_mean);
	auto offset1 = std::size_t(image_size.area());
	auto offset2 = std::size_t(2 * offset1);
#if 0
	auto p_data = (const float*)(image.data);
	for (std::size_t p = 0; p < offset1; p++) {
		dst[p] = *p_data++;
		dst[p + offset1] = *p_data++;
		dst[p + offset2] = *p_data++;
	}
#else
	int p = 0;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			dst[p] = pixel[0];
			dst[p + offset1] = pixel[1];
			dst[p + offset2] = pixel[2];
			p++;
		}
	}
#endif
}

void load_image_in_first_channel_format_with_heatmap(
	const std::wstring& imagePath, 
	const cv::Size& image_size, 
	const cv::Mat& heatMap,
	float intensityFactor,
	float* dst) {

	auto image = load_image_and_subtract_mean(imagePath, image_size, false);

	auto offset1 = std::size_t(image_size.area());
	auto offset2 = std::size_t(2 * offset1);

	// blend the two images
	auto max_value = 0.0f;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			auto& heatMapPixel = heatMap.at<cv::Vec3b>(row, col);
			pixel[0] += intensityFactor * (float)heatMapPixel[0];
			pixel[1] += intensityFactor * (float)heatMapPixel[1];
			pixel[2] += intensityFactor * (float)heatMapPixel[2];
			max_value = std::max(max_value, pixel[0]);
			max_value = std::max(max_value, pixel[1]);
			max_value = std::max(max_value, pixel[2]);
		}
	}

	// make sure we are in 0...255 range
	max_value /= 255;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			pixel[0] /= max_value;
			pixel[1] /= max_value;
			pixel[2] /= max_value;
		}
	}

	// finally copy it back
	int p = 0;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			dst[p] = pixel[0];
			dst[p + offset1] = pixel[1];
			dst[p + offset2] = pixel[2];
			p++;
		}
	}
}

cv::Mat load_image_with_heatmap(
	const std::wstring& path,
	const cv::Size& image_size,
	const cv::Mat& heatMap,
	float intensityFactor) {

	std::ifstream stream(path, std::ios::in | std::ios::binary);
	std::vector<uint8_t> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
	cv::Mat image(cv::Size(0, 0), CV_32FC3);
	cv::imdecode(contents, cv::IMREAD_COLOR, &image);
	cv::resize(image, image, image_size);
	image.convertTo(image, CV_32FC3);

	std::cout << "image.type()=" << image.type() << std::endl;
	cv::Mat nImage;
	image.convertTo(nImage, CV_32FC3, 1.0/255.);
	cv::imshow("the image", nImage);
	cv::waitKey();
	
	auto max_value = 0.0f;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			auto& heatMapPixel = heatMap.at<cv::Vec3b>(row, col);
			pixel[0] += intensityFactor * (float)heatMapPixel[0];
			pixel[1] += intensityFactor * (float)heatMapPixel[1];
			pixel[2] += intensityFactor * (float)heatMapPixel[2];
			max_value = std::max(max_value, pixel[0]);
			max_value = std::max(max_value, pixel[1]);
			max_value = std::max(max_value, pixel[2]);
		}
	}
	max_value = max_value / 255;
	for (int row = 0; row < image_size.height; row++) {
		for (int col = 0; col < image_size.width; col++) {
			auto& pixel = image.at<cv::Vec3f>(row, col);
			pixel[0] /= max_value;
			pixel[1] /= max_value;
			pixel[2] /= max_value;
		}
	}

	image.convertTo(nImage, CV_32FC3, 1.0 / 255.);
	cv::imshow("the image", nImage);
	cv::waitKey();

	return image;
}


void load_image(const wchar_t* imagePath, float* dst) {
	load_image_in_first_channel_format(imagePath, cv::Size(224, 224), dst, false);
}

CNTK::FunctionPtr vgg16(const wchar_t* modelPath, const CNTK::Variable& features, const wchar_t* layer_name, const CNTK::DeviceDescriptor& computeDevice) {
	auto model = CNTK::Function::Load(modelPath);
	auto last_frozen_layer = model->FindByName(layer_name);
	auto conv1_1_layer = model->FindByName(L"conv1_1");

	CNTK::Variable data;
	for (auto &v : conv1_1_layer->Inputs()) { if (v.Name() == L"data") { data = v; break; } }

	auto result = CNTK::Combine({ last_frozen_layer })->Clone(CNTK::ParameterCloningMethod::Freeze, { { data, features } });
	return result;
}

CNTK::DeviceDescriptor get_compute_device() {
	for (auto gpuDevice: CNTK::DeviceDescriptor::AllDevices()) {
		if (gpuDevice.Type() == CNTK::DeviceKind::GPU) { return gpuDevice; }
	}
	return CNTK::DeviceDescriptor::CPUDevice();
}


void compute_image(float* image, const wchar_t* pathToVGG16model, int filterIndex) {
	auto computeDevice = get_compute_device();

	auto layer_name = L"conv3_1";
	CNTK::NDShape shape({ 150, 150, 3 });
	auto features = CNTK::InputVariable(shape, CNTK::DataType::Float, /*needs_gradient*/ true, L"features");
	auto scaled_features = CNTK::ElementTimes(CNTK::Constant::Scalar((float)(1.0 / 255.0), computeDevice), features);

	auto model = vgg16(pathToVGG16model, scaled_features, layer_name, computeDevice);

	auto filter_0 = CNTK::Slice(model, { CNTK::Axis(2) }, { filterIndex }, { filterIndex+1 });
	auto loss_function = CNTK::ReduceMean(filter_0, CNTK::Axis::AllStaticAxes(), L"loss_function");
	auto conv1_1_layer = model->FindByName(L"conv1_1");
	for (auto &v : conv1_1_layer->Inputs()) { if (v.Name() == L"features") { features = v; break; } }

	std::vector<float> input_img_data(features.Shape().TotalSize());
	for (auto& v : input_img_data) { v = (float(128 + 20.0 * (rand() / double(RAND_MAX)))); };

	auto step = 1.0;
	const int numSteps = 40;
	for (int i = 0; i < numSteps; i++) {
		std::wcout << "Step " << (i+1) << "/" << numSteps << std::endl;
		auto batch = CNTK::Value::CreateBatch(shape, input_img_data, computeDevice);
		dict_t arguments{ { features, batch } };
		dict_t gradients{ { features, nullptr } };
		loss_function->Gradients(arguments, gradients, computeDevice);

		std::vector<std::vector<float>> batchResults;
		gradients[features]->CopyVariableValueTo(features, batchResults);
		assert(batchResults.size() == 1);

		auto& results = batchResults[0];
		assert(results.size() == input_img_data.size());
		double sum_squares = 0.0;
		for (auto v : results) { sum_squares += v*v; }
		auto mean_sum_squares = sum_squares / double(results.size());
		auto normalization_factor = float(sqrt(mean_sum_squares));
		for (auto&v : results) { v /= normalization_factor; }
		for (std::size_t i = 0; i < results.size(); ++i) {
			input_img_data[i] += float(step * results[i]);
		}
	}
	memcpy(image, &input_img_data[0], shape.TotalSize() * sizeof(float));
}


void evaluate_vgg16(const wchar_t* pathToVGG16model, const wchar_t* imagePath, float* predictions, int num_classes) {
	auto model = CNTK::Function::Load(pathToVGG16model);
	const std::size_t width = 224;
	const std::size_t height = 224;
	const std::size_t numChannels = 3;
	CNTK::NDShape shape{ height, width, numChannels };
	std::vector<float> image(width*height*numChannels);
	load_image_in_first_channel_format(imagePath, cv::Size(width, height), &image[0], true);

	auto computeDevice = get_compute_device();
	auto inputVariable = model->Arguments()[0];
	auto inputValue = CNTK::Value::CreateBatch(shape, image, computeDevice);
	dict_t inputMap = { { inputVariable, inputValue } };

	auto outputVariable = model->Output();
	dict_t outputMap = { { outputVariable, nullptr } };

	model->Evaluate(inputMap, outputMap, computeDevice);
	auto outputValue = outputMap[outputVariable];
	std::vector<std::vector<float>> outputData;
	outputValue->CopyVariableValueTo(outputVariable, outputData);
	memcpy(predictions, &outputData[0][0], num_classes * sizeof(float));
}

class Indexer3D {
public:
	Indexer3D(const float* values, const std::size_t numRows, const std::size_t numColumns, const std::size_t numChannels) :
		_values(values), _numRows(numRows), _numColumns(numColumns), _numChannels(numChannels), _area(_numRows*_numColumns) {}
	
	float at(const std::size_t row, const std::size_t column, const std::size_t channel) {
		auto index = channel * _area + row * _numColumns + column;
		return _values[index];
	}
protected:
	const float* _values;
	const std::size_t _numRows;
	const std::size_t _numColumns;
	const std::size_t _numChannels;
	const std::size_t _area;
};

void visualize_heatmap(const wchar_t* pathToVGG16model,
	const wchar_t* imagePath,
	const wchar_t* layerName,
	int predictionIndex,
	float* imageWithOverlayedHitmap) {

	// STEP 0: load image and model
	auto computeDevice = get_compute_device();
	auto model = CNTK::Function::Load(pathToVGG16model);
	const std::size_t width = 224;
	const std::size_t height = 224;
	const std::size_t numChannels = 3;
	CNTK::NDShape shape{ height, width, numChannels };
	std::vector<float> image(width*height*numChannels);
	load_image_in_first_channel_format(imagePath, cv::Size(width, height), &image[0], true);

	// STEP 1: break the model into two parts
	auto layer = model->FindByName(layerName);
	auto topModelInputTensor = CNTK::InputVariable(layer->Output().Shape(), CNTK::DataType::Float, true /*needs_gradient */, L"topModelData");
	std::unordered_map<CNTK::Variable, CNTK::Variable> replacements = { { layer, topModelInputTensor } };	
	auto top_model = CNTK::Combine({model})->Clone(CNTK::ParameterCloningMethod::Freeze, replacements);
	auto bottom_model = CNTK::Combine({ layer })->Clone(CNTK::ParameterCloningMethod::Freeze);
	auto topModelTargetPrediction = CNTK::Slice(top_model->Output(), { CNTK::Axis(0) }, { predictionIndex }, { predictionIndex + 1 });

	// STEP 2: get the bottom layer predictions
	dict_t bottomModelInputMap  { { bottom_model->Arguments()[0], CNTK::Value::CreateBatch(shape, image, computeDevice)} };
	dict_t bottomModelOutputMap { {bottom_model->Output(), nullptr} };
	bottom_model->Evaluate(bottomModelInputMap, bottomModelOutputMap, computeDevice);
	std::vector<std::vector<float>> bottomModelBatchPredictions;
	bottomModelOutputMap[bottom_model->Output()]->CopyVariableValueTo(bottom_model->Output(), bottomModelBatchPredictions);
	
	// STEP 3: compute the gradients
	dict_t gradientsInputMap { {topModelInputTensor, CNTK::Value::CreateBatch(topModelInputTensor.Shape(), bottomModelBatchPredictions[0], computeDevice) } };
	dict_t gradientsOutputMap{ { topModelInputTensor, nullptr } };
	dict_t debugOutputMap { { topModelTargetPrediction->Output(), nullptr } };
	topModelTargetPrediction->Gradients(gradientsInputMap, gradientsOutputMap, debugOutputMap, computeDevice);

	std::vector<std::vector<float>> batchGradients;
	gradientsOutputMap[topModelInputTensor]->CopyVariableValueTo(topModelInputTensor, batchGradients);
	const auto& gradients = batchGradients[0];

	// STEP 4: compute pooled_gradients
	auto numFeatureChannels = topModelInputTensor.Shape()[2];
	std::vector<float> pooled_gradients(numFeatureChannels);
	auto gridArea = topModelInputTensor.Shape()[0] * topModelInputTensor.Shape()[1];
	for (std::size_t channel = 0; channel < numFeatureChannels; channel++) {
		pooled_gradients[channel] = std::accumulate(gradients.begin() + channel*gridArea, gradients.begin() + (channel + 1)*gridArea, 0.0f) / gridArea;
	}

	// STEP 5: adjust the bottom layer predictions with the pooled_gradients
	auto& bottomModelPredictions = bottomModelBatchPredictions[0];
	for (std::size_t channel = 0; channel < numFeatureChannels; channel++) {
		auto offset = channel*gridArea;
		for (auto j = 0; j < gridArea; j++) { bottomModelPredictions[j+offset] *= pooled_gradients[channel]; }	
	}

	// STEP 6: compute the heatmap
	std::vector<float> heatMap(gridArea);
	for (std::size_t j = 0; j < heatMap.size(); j++) {
		float sum = 0;
		for (auto channel = 0; channel < numFeatureChannels; channel++) {
			auto offset = channel*gridArea + j;
			sum += bottomModelPredictions[offset];
		}
		heatMap[j] = std::max(sum / numFeatureChannels, 0.0f);
	}
	auto maxValue = *std::max_element(heatMap.begin(), heatMap.end());

	for (auto& v : heatMap) { v /= maxValue; }

	std::vector<uint8_t> heatMap8(gridArea);
	for (std::size_t j = 0; j < gridArea; j++) {
		heatMap8[j] = (uint8_t)(255 * heatMap[j]);
	}

	// STEP 7: apply heatmap to original image using OpenCV
	cv::Size heatMapSize((int)topModelInputTensor.Shape()[0], (int)topModelInputTensor.Shape()[1]);	
	cv::Mat heatMapImage(heatMapSize, CV_8UC1, &heatMap8[0]);
	cv::resize(heatMapImage, heatMapImage, cv::Size(height, width));
	cv::Mat colorMap;
	cv::applyColorMap(heatMapImage, colorMap, cv::COLORMAP_JET);

	load_image_in_first_channel_format_with_heatmap(imagePath, cv::Size(width, height), colorMap, 0.4f, &image[0]);
	if (imageWithOverlayedHitmap != nullptr) {
		memcpy(imageWithOverlayedHitmap, &image[0], image.size() * sizeof(float));
	}
}

void debug() {
	auto a = version();
	std::wcout << L"over here;" << a << std::endl;

	auto modelPath = L"C:\\Users\\anastasios\\Documents\\deep-learning-with-cntk-and-csharp\\DeepLearning\\DownloadedModels\\VGG16_ImageNet_Caffe.model";
	auto imagePath = L"C:\\Users\\anastasios\\Desktop\\creative_commons_elephant.jpg";

	//auto num_classes = 1000;
	//std::vector<float> predictions(num_classes);
	//evaluate_vgg16(modelPath, imagePath, &predictions[0], num_classes);
	//std::cout << predictions[386] << std::endl;

	visualize_heatmap(modelPath, imagePath, L"conv5_3", 386, nullptr); 
}

void debug_1() {
	const std::size_t width = 224;
	const std::size_t height = 224;
	const std::size_t numChannels = 3;
	CNTK::NDShape shape{ height, width, numChannels };

	std::vector<float> image(width * height * 3);
#if 0
	auto filePath = L"C:\\Users\\anastasios\\Documents\\deep-learning-with-cntk-and-csharp\\Python\\flatImage.bin";
	load_bin(filePath, image);
#else
	auto img_path = L"C:\\Users\\anastasios\\Desktop\\creative_commons_elephant.jpg";
	load_image_in_first_channel_format(img_path, cv::Size(width, height), &image[0], true);
#endif

	auto modelPath = L"C:\\Users\\anastasios\\Documents\\deep-learning-with-cntk-and-csharp\\DeepLearning\\DownloadedModels\\VGG16_ImageNet_Caffe.model";
	auto model = CNTK::Function::Load(modelPath);

	auto computeDevice = get_compute_device();
	auto inputVariable = model->Arguments()[0];
	auto inputValue = CNTK::Value::CreateBatch(shape, image, computeDevice);
	dict_t inputMap = { { inputVariable, inputValue } };
	auto outputVariable = model->Output();
	dict_t outputMap = { { outputVariable, nullptr } };

	model->Evaluate(inputMap, outputMap, computeDevice);
	auto outputValue = outputMap[outputVariable];
	std::vector<std::vector<float>> outputData;
	outputValue->CopyVariableValueTo(outputVariable, outputData);
	
	auto& predictions = outputData[0];
	std::cout << "over here" << predictions[386] << std::endl;
}

