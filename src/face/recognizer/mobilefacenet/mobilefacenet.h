#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include "../recognizer.h"
#include <vector>
#include "ncnn/net.h"

namespace mirror {

class Mobilefacenet : public Recognizer {
public:
	Mobilefacenet();
	~Mobilefacenet();

	int LoadModel(const char* root_path);
	int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);

private:
	ncnn::Net* mobileface_net_;
	bool initialized_;
	// add normalization mean value hear by FANG
    const float meanVals[3] = {     127.5f,     127.5f,     127.5f };
    const float normVals[3] = { 0.0078125f, 0.0078125f, 0.0078125f };
};

}

#endif // !_FACE_MOBILEFACENET_H_

