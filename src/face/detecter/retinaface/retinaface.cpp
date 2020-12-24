#include "retinaface.h"
#include <iostream>

//#if MIRROR_VULKAN
//#include "gpu.h"
//#endif // MIRROR_VULKAN

namespace mirror {
RetinaFace::RetinaFace() :
    retina_net_(new ncnn::Net()),
    initialized_(false) {
#if MIRROR_VULKAN
    ncnn::create_gpu_instance();
    retina_net_->opt.use_vulkan_compute = true;
#endif // MIRROR_VULKAN
}
RetinaFace::~RetinaFace() {
    if (retina_net_) {
        retina_net_->clear();
    }
#if MIRROR_VULKAN
    ncnn::destroy_gpu_instance();
#endif // MIRROR_VULKAN
}
//RetinaFace::~RetinaFace(){
//    Release();
//}

inline void RetinaFace::Release(){
    if (retina_net_ != nullptr)
    {
        delete retina_net_;
        retina_net_ = nullptr;
    }
}

RetinaFace::RetinaFace(const std::string &model_param, const std::string &model_bin, bool retinaface):
        _nms(0.5),
        _threshold(0.5),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(retinaface),
        retina_net_(new ncnn::Net())
{
    Init(model_param, model_bin);
}

inline bool RetinaFace::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

//inline void RetinaFace::SetDefaultParams(){
//    _nms = 0.4;
//    _threshold = 0.6;
//    _mean_val[0] = 104;
//    _mean_val[1] = 117;
//    _mean_val[2] = 123;
//    retina_net_ = nullptr;

//}



void RetinaFace::Init(const std::string &model_param, const std::string &model_bin)
{
    int ret = retina_net_->load_param(model_param.c_str());
    ret = retina_net_->load_model(model_bin.c_str());
}

void RetinaFace::create_anchor(std::vector<box> &anchor, int w, int h)
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

void RetinaFace::create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
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

void RetinaFace::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
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


void RetinaFace::Detect(const cv::Mat& img_src, std::vector<bbox>& faces)
{
//    float _mean_val{104.f, 117.f, 123.f};

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, img_src.cols, img_src.rows);
    in.substract_mean_normalize(_mean_val, 0);

    ncnn::Extractor ex = retina_net_->create_extractor();
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
    std::cout<<total_box.size()<<std::endl;
    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms=(0.5));
    std::cout<<total_box.size()<<_nms<<std::endl;
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
        faces.push_back(total_box[maxloc.x]);
        std::cout<<"face boxes------"<<faces[0].s<<std::endl;

    }

}

int RetinaFace::LoadModel(const char *root_path) {
    std::string fd_param =  std::string(root_path)+"/face.param";
    std::string fd_bin =std::string(root_path)+ "/face.bin";
//    RetinaFace torss;
//    torss.Init(fd_param,fd_bin);
    if (retina_net_->load_param(fd_param.c_str()) == -1 ||
        retina_net_->load_model(fd_bin.c_str()) == -1) {
        std::cout << "load face detect model failed." << std::endl;
        return 10000;
    }

//	// generate anchors
//	for (int i = 0; i < 3; ++i) {
//		ANCHORS anchors;
//		if (0 == i) {
//			GenerateAnchors(16, { 1.0f }, { 32, 16 }, &anchors);
//		}
//		else if (1 == i) {
//			GenerateAnchors(16, { 1.0f }, { 8, 4 }, &anchors);
//		}
//		else {
//			GenerateAnchors(16, { 1.0f }, { 2, 1 }, &anchors);
//		}
//		anchors_generated_.push_back(anchors);
//	}
    initialized_ = true;

//    return 0;
}
int RetinaFace::DetectFace(const cv::Mat & img_src,std::vector<bbox>& faces) {
    std::cout << "start face detect." << std::endl;
    faces.clear();
    if (!initialized_) {
        std::cout << "retinaface detector model uninitialized." << std::endl;
        return 10000;
    }
    if (img_src.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }
//    std::vector<bbox> boxes;
    RetinaFace::Detect(img_src, faces);
//    return boxes;
/*    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR, img_src.cols, img_src.rows, img_src.cols, img_src.rows);
    in.substract_mean_normalize(_mean_val, 0);

    ncnn::Extractor ex = retina_net_->create_extractor();
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
    }*/
}
//int RetinaFace::DetectFace(const cv::Mat & img_src,
//    std::vector<FaceInfo>* faces) {
//    std::cout << "start face detect." << std::endl;
//    faces->clear();
//    if (!initialized_) {
//        std::cout << "retinaface detector model uninitialized." << std::endl;
//        return 10000;
//    }
//    if (img_src.empty()) {
//        std::cout << "input empty." << std::endl;
//        return 10001;
//    }
//    cv::Mat img_cpy = img_src.clone();
//    int img_width = img_cpy.cols;
//    int img_height = img_cpy.rows;
//    float factor_x = static_cast<float>(img_width) / inputSize_.width;
//    float factor_y = static_cast<float>(img_height) / inputSize_.height;
//    ncnn::Extractor ex = retina_net_->create_extractor();
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
//        ncnn::Mat::PIXEL_BGR2RGB, img_width, img_height, inputSize_.width, inputSize_.height);
//    ex.input("data", in);

//    std::vector<FaceInfo> faces_tmp;
//    for (int i = 0; i < 3; ++i) {
//        std::string class_layer_name = "face_rpn_cls_prob_reshape_stride" + std::to_string(RPNs_[i]);
//        std::string bbox_layer_name = "face_rpn_bbox_pred_stride" + std::to_string(RPNs_[i]);
//        std::string landmark_layer_name = "face_rpn_landmark_pred_stride" + std::to_string(RPNs_[i]);

//        ncnn::Mat class_mat, bbox_mat, landmark_mat;
//        ex.extract(class_layer_name.c_str(), class_mat);
//        ex.extract(bbox_layer_name.c_str(), bbox_mat);
//        ex.extract(landmark_layer_name.c_str(), landmark_mat);

//        ANCHORS anchors = anchors_generated_.at(i);
//        int width = class_mat.w;
//        int height = class_mat.h;
//        int anchor_num = static_cast<int>(anchors.size());
//        for (int h = 0; h < height; ++h) {
//            for (int w = 0; w < width; ++w) {
//                int index = h * width + w;
//                for (int a = 0; a < anchor_num; ++a) {
//                    float score = class_mat.channel(anchor_num + a)[index];
//                    if (score < scoreThreshold_) {
//                        continue;
//                    }
//                    // 1.获取anchor生成的box
//                    cv::Rect box = cv::Rect(w * RPNs_[i] + anchors[a].x,
//                        h * RPNs_[i] + anchors[a].y,
//                        anchors[a].width,
//                        anchors[a].height);

//                    // 2.解析出偏移量
//                    float delta_x = bbox_mat.channel(a * 4 + 0)[index];
//                    float delta_y = bbox_mat.channel(a * 4 + 1)[index];
//                    float delta_w = bbox_mat.channel(a * 4 + 2)[index];
//                    float delta_h = bbox_mat.channel(a * 4 + 3)[index];

//                    // 3.计算anchor box的中心
//                    cv::Point2f center = cv::Point2f(box.x + box.width * 0.5f,
//                        box.y + box.height * 0.5f);

//                    // 4.计算框的实际中心（anchor的中心+偏移量）
//                    center.x = center.x + delta_x * box.width;
//                    center.y = center.y + delta_y * box.height;

//                    // 5.计算出实际的宽和高
//                    float curr_width = std::exp(delta_w) * (box.width + 1);
//                    float curr_height = std::exp(delta_h) * (box.height + 1);

//                    // 6.获取实际的矩形位置
//                    cv::Rect curr_box = cv::Rect(center.x - curr_width * 0.5f,
//                        center.y - curr_height * 0.5f, curr_width, 	curr_height);
//                    curr_box.x = MAX(curr_box.x * factor_x, 0);
//                    curr_box.y = MAX(curr_box.y * factor_y, 0);
//                    curr_box.width = MIN(img_width - curr_box.x, curr_box.width * factor_x);
//                    curr_box.height = MIN(img_height - curr_box.y, curr_box.height * factor_y);

//                    FaceInfo face_info;
//                    memset(&face_info, 0, sizeof(face_info));

//                    int offset_index = landmark_mat.c / anchor_num;
//                    for (int k = 0; k < 5; ++k) {
//                        float x = landmark_mat.channel(a * offset_index + 2 * k)[index] * box.width + center.x;
//                        float y = landmark_mat.channel(a * offset_index + 2 * k + 1)[index] * box.height + center.y;
//                        face_info.keypoints_[k] = MIN(MAX(x * factor_x, 0.0f), img_width - 1);
//                        face_info.keypoints_[k + 5] = MIN(MAX(y * factor_y, 0.0f), img_height - 1);
//                    }

//                    face_info.score_ = score;
//                    face_info.location_ = curr_box;
//                    faces_tmp.push_back(face_info);
//                }
//            }
//        }
//    }

//    NMS(faces_tmp, faces, iouThreshold_);
//    std::cout << faces->size() << " faces detected." << std::endl;

//    std::cout << "end face detect." << std::endl;
//    return 0;
//}

}

