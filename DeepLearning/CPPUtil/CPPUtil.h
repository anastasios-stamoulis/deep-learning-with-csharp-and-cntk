#pragma once

#ifdef CPP_EXPERIMENT
#define DLL_API
#elif CPPUTIL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

extern "C" {
	DLL_API double version();
	DLL_API void compute_image(float* image, const wchar_t* pathToVGG16model, int filterIndex);
	DLL_API void load_image(const wchar_t* imagePath, float* image);
	DLL_API void evaluate_vgg16(const wchar_t* pathToVGG16model, const wchar_t* imagePath, float* predictions, int num_classes);
	DLL_API void visualize_heatmap(const wchar_t* pathToVGG16model, const wchar_t* imagePath, const wchar_t* layerName, int predictionIndex, float* imageWithOverlayedHitmap);
}

