#define FACE_EXPORTS
#include <typeinfo>
#include "opencv2/opencv.hpp"
#include "face_engine.h"
#include "ncnn/net.h"

using namespace mirror;

std::string detect_mobilemasknet(cv::Mat& bgr,ncnn::Extractor &netss)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows,28, 28);
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float e_vals[3] = { 1.0f/128.0f,1.0f/128.0f,1.0f/128.0f };
    in.substract_mean_normalize(mean_vals, e_vals);
    ncnn::Mat out;
    netss.input("input.1", in);
    netss.extract("522", out);

    std::string labels[2]={"mask","no_mask"};
    std::string outstring="";
//    std::cout<<"out[0]"<<out[0]<<" out[1]"<<out[1]<<std::endl;
    if (out[0]>out[1]){
        outstring=labels[0];

    }
    else {
        outstring=labels[1];
    }
    return outstring;
}



cv::Mat face_alignment(cv::Mat &img, std::vector<cv::Point2f> landmarks)
{
    std::vector<cv::Point2f> p2s;
    p2s.push_back(cv::Point2f(30.2946, 51.6963));
    p2s.push_back(cv::Point2f(65.5318, 51.5014));
    p2s.push_back(cv::Point2f(48.0252, 71.7366));
    p2s.push_back(cv::Point2f(33.5493, 92.3655));
    p2s.push_back(cv::Point2f(62.7299, 92.2041));

    cv::Mat ret;
    std::vector<cv::Point2f> p1s;
    for (int j = 0; j < 5; j++)
    {
        p1s.push_back(cv::Point(landmarks[j].x, landmarks[j].y));
        // std::cout << cv::Point(landmarks[j].x, landmarks[j].y) << std::endl;
    }
    cv::Mat rot_mat = cv::estimateAffinePartial2D(p1s, p2s);
    std::cout<<"---rot mat"<<rot_mat<<std::endl;
    cv::warpAffine(img, ret, rot_mat, cv::Size(112, 112));

    return ret;
}

static int face_Regc(const cv::Mat& bgr,std::vector<float>* feature)
{
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("../../data/models/mbface_nobn_mask.param");
    mobilefacenet.load_model("../../data/models/mbface_nobn_mask.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows,112, 112);

    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float e_vals[3] = { 1.0f/128.0f,1.0f/128.0f,1.0f/128.0f };
    in.substract_mean_normalize(mean_vals, e_vals);

    ncnn::Extractor ex = mobilefacenet.create_extractor();

    ex.input("input0", in);

    ncnn::Mat out;
    ex.extract("output1", out);
    feature->resize(128);
    for (int i = 0; i < 128; ++i) {
        feature->at(i) = out[i];
    }

    return 0;
}



int TestVideo(int argc, char* argv[]){
    const char* root_path = "../../data/models";
    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);

    std::string save_impath="../../data/images/mask_data/save_imgs";

    //input video test
    int caps_id=0;
    std::cout<<"请选择摄像头ID :: 默认ID是 "<<caps_id<<std::endl;
    std::cin>>caps_id;
    cv::VideoCapture cap(caps_id);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);//宽度
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);//高度
    if (!cap.isOpened())
        {
        std::cout << "Unable to connect to camera!" << std::endl;
        return EXIT_FAILURE;
        }
    int s_size=2.0;
    /*base face image*/
    const char* face_path = "../../data/images/face_base/bb_face2.jpg";
    const char* face_path2 = "../../data/images/face_base/zc2.png";
    cv::Mat face1=cv::imread(face_path2);
//    cv::Mat facezc=cv::imread(face_path2);

    /* if use source face image need detection_Face and align
    std::vector<bbox> face_box;
    face_engine->DetectFace(face1, face_box);
    int num_faces1 = static_cast<int>(face_box.size());
    for (int i = 0; i < num_faces1; ++i) {

        float left = face_box[i].x1;
        float top = face_box[i].y1;
        float right = face_box[i].x2;
        float bottom = face_box[i].y2;
        float conf = face_box[i].s;
        float width = right - left;
        float height = bottom - top;
        cv::Rect face(left,top,width,height);
        cv::Mat face_img=face1(face);

        align ----->Face-----
        std::vector<cv::Point2f> curr_pt1;
        for (int num = 0; num < 5; ++num) {
            curr_pt1.push_back(cv::Point2f(face_box[i].point[num]._x, face_box[i].point[num]._y));
//            cv::circle(face1, curr_pt1, 2, cv::Scalar(0, 0, 255), 2);
        }
        cv::Mat faceb=face_alignment(face1,curr_pt1);
        cv::Mat face_aligned1;
        face_engine->AlignFace(face1, curr_pt1, &face_aligned1);
        cv::imshow("aligned1",face_aligned1);
        cv::waitKey(1);
        cv::imshow("imgs",face_img);
        cv::waitKey(1);
        curr_pt1.clear();
    }*/

    /* init masknet*/
    ncnn::Net mobilemasknet;
    mobilemasknet.load_param("../../data/models/mb28-maskclassifier.param");
    mobilemasknet.load_model("../../data/models/mb28-maskclassifier.bin");

   /* init face regin*/

    std::vector<float> feat1;
    face_Regc(face1, &feat1);

    cv::imshow("imgss",face1);
    cv::waitKey(0);
    int frame_id=0;
    while(true)
        {
        cv::Mat img;
        cap >> img;
        frame_id+=1;
        if (img.empty()) {return -1;}
        cv::Mat img_copy = img.clone();
        cv::resize(img_copy, img_copy, cv::Size(img.cols/s_size, img.rows/s_size));
        std::vector<bbox> faces;
        double start1 = static_cast<double>(cv::getTickCount());
        face_engine->DetectFace(img_copy, faces);
        double end = static_cast<double>(cv::getTickCount());
        double detec_time_cost = (end - start1) / cv::getTickFrequency() * 1000;
        std::cout << "Detection time cost: " << detec_time_cost << "ms" << std::endl;
        int num_faces = static_cast<int>(faces.size());
        for (int i = 0; i < num_faces; ++i) {
            float left = faces[i].x1*s_size;
            float top = faces[i].y1*s_size;
            float right = faces[i].x2*s_size;
            float bottom = faces[i].y2*s_size;
            float conf = faces[i].s;
            float width = right - left;
            float height = bottom - top;
            cv::Rect face(left,top,width,height);
            cv::Mat face_img=img(face);
            cv::imshow("face_img",face_img);
            cv::waitKey(1);
            //align ----->Face-----
            std::vector<cv::Point2f> curr_pt;
            for (int num = 0; num < 5; ++num) {
                curr_pt.push_back(cv::Point2f(faces[i].point[num]._x*s_size, faces[i].point[num]._y*s_size));
            }
//            cv::Mat faceb=face_alignment(face1,curr_pt);
            cv::Mat face_aligned;
            face_engine->AlignFace(img, curr_pt, &face_aligned);
            std::vector<float> featdet;
            double start3 = static_cast<double>(cv::getTickCount());
            face_Regc(face_img,&featdet);
            float sim = CalculateSimilarity(feat1, featdet);
            double end3 = static_cast<double>(cv::getTickCount());
            std::cout << "similarity is: " << sim << std::endl;
            //人脸识别时间统计
            double Face_recongine_cost_time = (end3 - start3) / cv::getTickFrequency() * 1000;
            std::cout << "Face_recongine time cost: " << Face_recongine_cost_time << "ms" << std::endl;
            cv::imshow("aligned_img",face_aligned);
            cv::waitKey(1);
//            if (frame_id%4==0){
//                std::string imname=save_impath+"/"+std::to_string(frame_id)+".jpg";
//                std::cout<<"-------------"<<imname<<std::endl;
//                cv::imwrite(imname,face_img);}
            ncnn::Extractor nets = mobilemasknet.create_extractor();
            std::string output_mask=detect_mobilemasknet(face_aligned,nets);
            std::cout<<"---maskdetection---"<<output_mask<<std::endl;

            for (int j = 0; j < 5; ++j) {
                cv::circle(img,curr_pt[j],2,cv::Scalar(255,0,255),s_size);
            }

            cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2, 8, 0);
            if (sim>0.4)
                cv::putText(img, "zhangchao", cv::Point(int(face.x), int(face.y)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2, 8);
            else
                cv::putText(img, "unkonw", cv::Point(int(face.x), int(face.y)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, 8);

            cv::rectangle(img, face, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("result", img);
        cv::waitKey(1);
        }
    delete face_engine;
    face_engine = nullptr;
    std::cin.clear();
}

int TestRecognizePair(int argc, char* argv[]) {
    const char* root_path = "../../data/models";

    // const char* img_file1 = "../../data/images/fbb.jpg";
    // cv::Mat img_src1 = cv::imread(img_file1);

    // const char* img_file2 = "../../data/images/fbb1.jpg";
    // cv::Mat img_src2 = cv::imread(img_file2);

    const char* face1_path = "../../data/images/lz2.jpg";
    const char* face2_path = "../../data/images/bb_face2.jpg";
    // time cost: 4.940ms
    // similarity is: 0.743621
    // const char* face2_path = "../../data/images/face3.jpg";
    cv::Mat face1 = cv::imread(face1_path);
    cv::Mat face2 = cv::imread(face2_path);

    // double start = static_cast<double>(cv::getTickCount());
    FaceEngine* face_engine = new FaceEngine();
    face_engine->LoadModel(root_path);
    // std::vector<FaceInfo> faces1;
    // std::vector<FaceInfo> faces2;
    // face_engine->DetectFace(img_src1, &faces1);
    // face_engine->DetectFace(img_src2, &faces2);

    // cv::Mat face1 = img_src1(faces1[0].location_).clone();
    // cv::Mat face2 = img_src2(faces2[0].location_).clone();

    double start = static_cast<double>(cv::getTickCount());
    std::vector<float> feat1, feat2;
    face_engine->ExtractFeature(face1, &feat1);
    // double end = static_cast<double>(cv::getTickCount());
    face_engine->ExtractFeature(face2, &feat2);

    for (int x = 0; x !=1000; ++x)
    {
        std::vector<float> feat1, feat2;
        face_engine->ExtractFeature(face1, &feat1);
        face_engine->ExtractFeature(face2, &feat2);
        float sim = CalculateSimilarity(feat1, feat2);
        std::cout << "similarity is: " << sim << std::endl;
    }

    double end = static_cast<double>(cv::getTickCount());
    // double time_cost = (end - start) / cv::getTickFrequency();
    double time_cost = (end - start) / cv::getTickFrequency() * 1002;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;
    float sim = CalculateSimilarity(feat1, feat2);
    std::cout << "similarity is: " << sim << std::endl;

    delete face_engine;
    face_engine = nullptr;
    return 0;

}

int main(int argc, char* argv[]) {

    return TestVideo(argc,argv);}
