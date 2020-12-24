#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include "../detecter.h"
#include "ncnn/net.h"
/*
namespace mirror {
using ANCHORS = std::vector<cv::Rect>;
class RetinaFace : public Detecter {
public:
	RetinaFace();
	~RetinaFace();
	int LoadModel(const char* root_path);
	int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
	ncnn::Net* retina_net_;
	std::vector<ANCHORS> anchors_generated_;
	bool initialized_;
	const int RPNs_[3] = { 32, 16, 8 };
	const cv::Size inputSize_ = { 300, 300 };
	const float iouThreshold_ = 0.4f;
	const float scoreThreshold_ = 0.8f;

};
}
*/



namespace mirror {
using ANCHORS = std::vector<cv::Rect>;
class RetinaFace : public Detecter {
public:
    RetinaFace();
    void Init(const std::string &model_param, const std::string &model_bin);

    RetinaFace(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    void Detect(const cv::Mat & img_src, std::vector<bbox>& boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    int LoadModel(const char* root_path);
    int DetectFace(const cv::Mat& img_src, std::vector<bbox>& faces);

    ~RetinaFace();
public:
    float _nms=0.5;
    float _threshold=0.5;
    float _mean_val[3]={104.f, 117.f, 123.f};
    bool _retinaface;
    ncnn::Net* retina_net_;
//    ncnn::Net *Net;
private:
    std::vector<ANCHORS> anchors_generated_;
    bool initialized_;
    const int RPNs_[3] = { 32, 16, 8 };
    const cv::Size inputSize_ = { 300, 300 };
    const float iouThreshold_ = 0.4f;
    const float scoreThreshold_ = 0.8f;

};
}
/*
class Detectorss
{
public:
    Detectorss();

    void Init(const std::string &model_param, const std::string &model_bin);

    Detectorss(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    void Detect(const cv::Mat& img_src, std::vector<bbox>& boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    ~Detectorss();
public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;

    ncnn::Net *Net;
};

RetinaFace::~RetinaFace(){
    Release();
}
inline void Detectorss::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Detectorss::Detectorss(const std::string &model_param, const std::string &model_bin, bool retinaface):
        _nms(0.5),
        _threshold(0.5),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(retinaface),
        Net(new ncnn::Net())
{
    Init(model_param, model_bin);
}

inline bool Detectorss::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detectorss::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;

}



void Detectorss::Init(const std::string &model_param, const std::string &model_bin)
{
    int ret = Net->load_param(model_param.c_str());
    ret = Net->load_model(model_bin.c_str());
}

void Detectorss::create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detectorss::create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
    // anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    // TODO:
    for (int k = 0; k < feature_map.size(); ++k)  // 3
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detectorss::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


void Detectorss::Detect(const cv::Mat& img_src, std::vector<bbox>& boxes)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, img_src.cols, img_src.rows);
    in.substract_mean_normalize(_mean_val, 0);

    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);

    ncnn::Mat out, out1, out2;
    // loc
    ex.extract("output0", out);

    // class
    ex.extract("530", out1);

    //landmark
    ex.extract("529", out2);

    std::vector<box> anchor;
    if (_retinaface)
        create_anchor_retinaface(anchor,  img_src.cols, img_src.rows);
    else
        create_anchor(anchor,  img_src.cols, img_src.rows);

    // TODO: opencv dnn extract
    std::vector<bbox> total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.w;
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.h;
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }
//    std::cout<<'----------------'<<total_box<<std::endl;
    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    //select the max box
    if (total_box.size()>0){
        std::vector<int> area_boxes;
        for(int ii=0; ii < total_box.size(); ++ii){
            int area_box = (total_box[ii].x2 - total_box[ii].x1)*
                    (total_box[ii].y2 - total_box[ii].y1);
            area_boxes.push_back(area_box);
        }
        cv::Point maxloc;
        std:double maxvaule;
        cv::minMaxLoc(area_boxes, NULL, &maxvaule, NULL, &maxloc);
        boxes.push_back(total_box[maxloc.x]);
    }  //Created on Sep 09, 2020 by Qiao
}
*/
#endif // !_RETINAFACE_H_

