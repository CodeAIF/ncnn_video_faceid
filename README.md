# **ncnn_example:**
# add retinaface facedetection;add face_id;add mask_classify
## 2.optimize the CMakeLists.txt
## 3. compile the project:
```
>> cd ncnn_video_faceid && mkdir build && cd build && make -j3 
```
## 4. run the project:
```
>> cd src && ./face && ./object && ./classifier
```
# 7. references
## https://github.com/Tencent/ncnn
## https:/github.com/MirrorYuChen/ncnn_example/
## https://github.com/Charrin/RetinaFace-Cpp
## https://github.com/deepinsight/insightface
## https://github.com/Star-Clouds/centerface
## https://github.com/seetafaceengine/SeetaFace2
