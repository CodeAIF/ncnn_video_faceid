#define CLASSIFIER_EXPORTS
#include <math.h>
#include <glob.h>
#include "face_engine.h"
#include <typeinfo>
#include<numeric>
#include <fstream>
#include "ncnn/net.h"
//#include "classifier_engine.h"
#include "opencv2/opencv.hpp"

double myfunction(double num) {
    return exp(num);
}

float CalculateSimilarity(const std::vector<float>&feature1, const std::vector<float>& feature2) {
    if (feature1.size() != feature2.size()) {
        std::cout << "feature size not match." << std::endl;
        return 10003;
    }
    float inner_product = 0.0f;
    float feature_norm1 = 0.0f;
    float feature_norm2 = 0.0f;
    for(int i = 0; i < kFaceFeatureDim; ++i) {
        inner_product += feature1[i] * feature2[i];
        feature_norm1 += feature1[i] * feature1[i];
        feature_norm2 += feature2[i] * feature2[i];
    }
    return inner_product / sqrt(feature_norm1) / sqrt(feature_norm2);
}
//void getImages(std::string path, std::vector<std::string>& imagesList)
//{
//    intptr_t hFile = 0;
//    struct _finddata_t fileinfo;
//    std::string p;

//    hFile = _findfirst(p.assign(path).append("/*.jpg").c_str(), &fileinfo);

//    if (hFile != -1) {
//        do {
//            imagesList.push_back(fileinfo.name);//保存类名
//            } while (_findnext(hFile, &fileinfo) == 0);
//    }
//}

static std::string detect_mobilemasknet(const cv::Mat& bgr,ncnn::Extractor &netss,std::vector<float>& cls_scores)
{


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows,28, 28);
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float e_vals[3] = { 1.0f/128.0f,1.0f/128.0f,1.0f/128.0f };
    in.substract_mean_normalize(mean_vals, e_vals);

//    ncnn::Extractor ex = mobilemasknets.create_extractor();

    netss.input("input.1", in);

    ncnn::Mat out;
    netss.extract("522", out);
//    std::cout<<"outs_type"<<typeid(out).name()<<"out--shape"<<out.h<<std::endl;
//    cls_scores.resize(out.w);
//    cv::Mat outs1[1][2];
//    for (int j = 0; j<out.w; j++)
//    {
//        cls_scores[j] = out[j];
//        outs1[1][j]=out[j];
//    }
    std::string labels[2]={"no_mask","mask"};
    std::string outstring="";
    std::cout<<"out[0]"<<out[0]<<" out[1]"<<out[1]<<std::endl;
    if (out[0]>out[1]){
        outstring=labels[0];
    }
    else {
        outstring=labels[1];
    }
    return outstring;
}

static int face_Regc(const cv::Mat& bgr,std::vector<float>* feature)
{
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("../../data/models/mbfacev2.param");
    mobilefacenet.load_model("../../data/models/mbfacev2.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows,112, 112);

    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float e_vals[3] = { 1.0f/128.0f,1.0f/128.0f,1.0f/128.0f };
    in.substract_mean_normalize(mean_vals, e_vals);

    ncnn::Extractor ex = mobilefacenet.create_extractor();

    ex.input("input.1", in);

    ncnn::Mat out;
    ex.extract("511", out);
    feature->resize(128);
    for (int i = 0; i < 128; ++i) {
        feature->at(i) = out[i];
    }

    return 0;
}


int main(int argc, char* argv[]) {
    const char* img_paths = "../../data/images/zc1.png";
    std::string impath="../../data/images/mask_data/save_imgs";
    std::vector<cv::Mat> images;
    cv::Mat img_src2 = cv::imread(img_paths);
    std::vector<float> feature2;
    face_Regc(img_src2,&feature2);
//    for (int i;i<128;++i){
//        std::cout<<feature2.at(i)<<std::endl;

//    }
    std::vector<cv::String> fbs;
    cv::glob(impath,fbs,false);
    size_t counts = fbs.size();
    if (counts==0)
    {
        std::cout << "file " << impath << " not  exits"<<std::endl;
        return -1;
    }
    for (int i ;i<counts;++i){
        std::cout<<"--imanames--"<<fbs[i]<<std::endl;
        cv::Mat img_src = cv::imread(fbs[i]);
        std::vector<float> feature1;

        face_Regc(img_src,& feature1);
        for (int i;i<128;++i){
            std::cout<<feature1.at(i)<<std::endl;
        }
        float sim = CalculateSimilarity(feature1, feature2);
        std::cout<<"--------------"<<sim<<std::endl;
        cv::imshow("imgss",img_src);
        cv::imshow("imgss",img_src2);
        cv::waitKey(0);
        std::vector<float>cls_scores;
        ncnn::Net mobilemasknet;
        mobilemasknet.load_param("../../data/models/mb28-maskclassifier.param");
        mobilemasknet.load_model("../../data/models/mb28-maskclassifier.bin");
        ncnn::Extractor nets = mobilemasknet.create_extractor();
        std::string output=detect_mobilemasknet(img_src,nets,cls_scores);
        cv::putText(img_src, output, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 10);
        std::cout<<"cls_Scores _sizes_outss  "<<output<<std::endl;
        cv::imshow("imgss",img_src);
        cv::waitKey(0);
        }
	return 0;
}

