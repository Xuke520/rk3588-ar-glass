#ifndef INVOKE_PYTHON_H
#define INVOKE_PYTHON_H

#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> 
#include <vector>



/**
 * 在mediapipe中手部坐标如下
 * 拇指:{0,1},{1,2},{2,3},{3,4}       
 * 食指:{0,5},{5,6},{6,7},{7,8}       
 * 中指:{9,10},{10,11},{11,12}        
 * 无名指:{13,14},{14,15},{15,16}        
 * 小指:{0,17},{17,18},{18,19},{19,20} 
 * 手掌基部:{5,9},{9,13},{13,17}            
 */
    
/**
 * 在实际的glVertices中对应的手势坐标如下*3
 * 目前最多支持两只手
 * 
 */
const std::vector<std::pair<int,int>> hand_connections={
    //第一只手
    {0,3},{3,6},{6,9},{9,12},        //拇指
    {0,15},{15,18},{18,21},{21,24},  //食指
    {27,30},{30,33},{33,36},         //中指
    {39,42},{42,45},{45,48},        //无名指
    {0,51},{51,54},{54,57},{57,60}, //小指
    {15,27},{27,39},{39,51},         //手掌基部
    //第二只手+63
    {63,66},{66,69},{69,72},{72,75},        //拇指
    {63,78},{78,81},{81,84},{84,87},  //食指
    {90,93},{93,96},{96,99},         //中指
    {102,105},{105,108},{108,111},        //无名指
    {63,114},{114,117},{117,120},{120,123}, //小指
    {78,90},{90,102},{102,114}         //手掌基部

};


struct Landmark{
    float x,y,z;
};

class SharedMemory{
public:
    std::vector<std::vector<cv::Point3f>> _landmarks;

    SharedMemory(const std::string& name)
    :_shm_name(name),_fd(-1),_ptr(nullptr),_size(0){open();}
    ~SharedMemory(){close();}
    bool read_cv_landmarks(std::vector<std::vector<cv::Point3f>>& hands);
    bool read_gl_landmarks(std::vector<float>& glVertices,float ratio=1.0f);

private:
    std::string _shm_name;
    int _fd;
    void* _ptr;
    size_t _size;

    bool open();
    void close();
    bool check_data();

};

class KalmanFilter{
private:
    float _process_noise;
    float _measure_noise;
    int _max_hands;
    int _max_predict_frames;
    int _count;
    std::vector<std::vector<cv::KalmanFilter>> _kfs;
    std::vector<std::vector<cv::Point3f>> _lfiltered_hands;
    std::vector<float> _lfiltered_glVertices;
    void create();
public:
    KalmanFilter(float process_noise=0.01f,float measure_noise=0.5f,int max_hands=1,int max_predict_frames=10);
    std::vector<std::vector<cv::Point3f>> run_cv(const std::vector<std::vector<cv::Point3f>>& hands,float dt);
    std::vector<float> run_gl(const std::vector<float>& glVertices,float dt);

};



std::vector<float> generate_line_vertices(const std::vector<float>& glVertices);



#endif