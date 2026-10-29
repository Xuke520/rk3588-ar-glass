#include "invoke_python.h"


bool SharedMemory::open()
{   
    /**
     * shm是linux的系统调用,用于管理共享内存,位置在/dev/shm/
     * shm_open是c函数,为了在c++中正常调用,使用::shm_open
     * ::用于指定全局命名空间中的shm_open函数,以避免与类或命名空间中的同名函数冲突
     * 
     */
    _fd=::shm_open(_shm_name.c_str(),O_RDONLY,0666);
    if(_fd<0){
        std::cout<<"shm_open failed"<<std::endl;
        return false;
    }
    /**
     * 获取共享内存大小
     * struct stat是一个C结构体,用于存储文件或共享内存对象的状态信息(如大小、权限、所有者)
     */
    struct stat sb;
    if(::fstat(_fd,&sb)==-1){
        std::cout<<"fstat failed"<<std::endl;
        return false;
    }
    _size=sb.st_size;

    _ptr=::mmap(nullptr,_size,PROT_READ,MAP_SHARED,_fd,0);
    if(_ptr==MAP_FAILED){
        std::cout<<"mmap failed"<<std::endl;
        return false;
    }
    return true;
}

void SharedMemory::close()
{
    if(_ptr!=nullptr&&_ptr!=MAP_FAILED){
        ::munmap(_ptr,_size);
    }
    if(_fd>=0){
        ::close(_fd);
    }
}

/**
 * @brief 验证共享内存中的内容是否符合预期格式(数据总量(4字节)+具体手势坐标点数据)
 * @return 数据正确返回true,数据错误或空返回false
 * 
 */
bool SharedMemory::check_data()
{
    if(_ptr==nullptr){
        std::cout<<"mmap failed"<<std::endl;
        return false;
    }
    /* 解析共享内存前四字节数据(对应有效数据的长度) */
    const uint32_t data_size=*static_cast<uint32_t*>(_ptr);
    if(data_size==0||data_size!=(_size-sizeof(uint32_t))){
        std::cout<<"no data or data size err"<<std::endl;
        return false;
    }else{
        /* 检查有效数据的长度是否是Landmark的整数倍 */
        if(data_size%sizeof(Landmark)!=0){
            std::cout<<"data size not multiple of landmark size"<<std::endl;
            return false;
        }
    }
    return true;
}


/**
 * @brief 读取共享内存中的手势坐标点数据,数据格式为:数据总量(4字节)+具体手势坐标点数据
 * @param hands 每个坐标点使用cv::Point3f表示,内容器存储一只手的21个坐标数据,外容器存储多只手
 * @return 读取到手势数据返回true,读取错误或空返回false
 * 
 */
bool SharedMemory::read_cv_landmarks(std::vector<std::vector<cv::Point3f>>& hands) 
{
    if(check_data()==false) return false;
    /* 读取数据 */
    const auto* data_ptr=reinterpret_cast<Landmark*>(static_cast<char*>(_ptr)+sizeof(uint32_t));
    const uint32_t data_size=*static_cast<uint32_t*>(_ptr);
    const int hands_num=(data_size/sizeof(Landmark))/21;
    for(int i=0;i<hands_num;i++){
        std::vector<cv::Point3f> singlehand_landmarks;
        for(int j=0;j<21;j++){
            const int idx=i*21+j;
            cv::Point3f landmark(data_ptr[idx].x,data_ptr[idx].y,data_ptr[idx].z);
            singlehand_landmarks.push_back(landmark);
        }
        hands.push_back(singlehand_landmarks);
    }
    return true;
}


bool SharedMemory::read_gl_landmarks(std::vector<float>& glVertices,float ratio) 
{
    if(check_data()==false) return false;
    /* 读取数据 */
    const auto* data_ptr=reinterpret_cast<Landmark*>(static_cast<char*>(_ptr)+sizeof(uint32_t));
    const uint32_t data_size=*static_cast<uint32_t*>(_ptr);
    const int hands_num=(data_size/sizeof(Landmark))/21;
    for(int i=0;i<hands_num;i++){
        for(int j=0;j<21;j++){
            const int idx=i*21+j;
            glVertices.push_back(ratio*(data_ptr[idx].x*2-1));
            glVertices.push_back(ratio*(1-2*data_ptr[idx].y));
            glVertices.push_back(data_ptr[idx].z);
        }
    }
    std::cout<<"glVertices====="<<glVertices.size()<<std::endl;
    return true;
}


KalmanFilter::KalmanFilter(float process_noise,float measure_noise,int max_hands,int max_predict_frames)
{
    _process_noise=process_noise;
    _measure_noise=measure_noise;
    _max_hands=max_hands;
    _max_predict_frames=max_predict_frames;
    _count=0;
    create();
}

void KalmanFilter::create()
{
    _kfs.clear();
    for(int i=0;i<_max_hands;i++){
        std::vector<cv::KalmanFilter> hand_kfs;
        for(int j=0;j<21;j++){
            cv::KalmanFilter kf(9,3,0);
            kf.transitionMatrix=(cv::Mat_<float>(9,9)<<
                1,0,0,1,0,0,0.5,0,0,
                0,1,0,0,1,0,0,0.5,0,
                0,0,1,0,0,1,0,0,0.5,
                0,0,0,1,0,0,1,0,0,
                0,0,0,0,1,0,0,1,0,
                0,0,0,0,0,1,0,0,1,
                0,0,0,0,0,0,1,0,0,
                0,0,0,0,0,0,0,1,0,
                0,0,0,0,0,0,0,0,1);
            // 测量矩阵
            kf.measurementMatrix=(cv::Mat_<float>(3,9)<<
                1,0,0,0,0,0,0,0,0,
                0,1,0,0,0,0,0,0,0,
                0,0,1,0,0,0,0,0,0);
            cv::setIdentity(kf.processNoiseCov,cv::Scalar::all(_process_noise));
            cv::setIdentity(kf.measurementNoiseCov,cv::Scalar::all(_measure_noise));
            kf.statePost=cv::Mat::zeros(9,1,CV_32F);
            cv::setIdentity(kf.errorCovPost,cv::Scalar::all(1));
            hand_kfs.push_back(kf);
        }
        _kfs.push_back(hand_kfs);
    }

}



/**
 * @brief 对手势坐标点数据进行卡尔曼滤波,点坐标格式为cv::Point3f
 *        当超过_max_predict_frames帧而没有检测到手势时清零,输出零矩阵
 * @param hands 共享内存中的手势坐标点数据
 * @return 返回滤波后的手势坐标点数据
 * 
 */
std::vector<std::vector<cv::Point3f>> KalmanFilter::run_cv(const std::vector<std::vector<cv::Point3f>>& hands,float dt) 
{
    std::vector<std::vector<cv::Point3f>> filtered_hands;

    if(!hands.empty()){ //检测到手部
        _count=0;
        for(int i=0;i<hands.size();i++){
            std::vector<cv::Point3f> current_hand;
            for(int j=0;j<21;j++){
                cv::KalmanFilter& kf=_kfs[i][j];

                // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
                // v(t+dt) = v(t) + a(t)*dt
                // a(t+dt) = a(t)

                kf.transitionMatrix=(cv::Mat_<float>(9,9)<<
                    1,0,0,dt,0,0,0.5*dt*dt,0,0,
                    0,1,0,0,dt,0,0,0.5*dt*dt,0,
                    0,0,1,0,0,dt,0,0,0.5*dt*dt,
                    0,0,0,1,0,0,dt,0,0,
                    0,0,0,0,1,0,0,dt,0,
                    0,0,0,0,0,1,0,0,dt,
                    0,0,0,0,0,0,1,0,0,
                    0,0,0,0,0,0,0,1,0,
                    0,0,0,0,0,0,0,0,1);
                kf.predict(); 
                /* 校正阶段 */
                cv::Mat measurement=(cv::Mat_<float>(3,1)<<hands[i][j].x,hands[i][j].y,hands[i][j].z);
                kf.correct(measurement);
                /* 获取结果 */
                cv::Mat state=kf.statePost;
                current_hand.emplace_back(state.at<float>(0),state.at<float>(1),state.at<float>(2));
            }
            filtered_hands.push_back(current_hand);
        }
        _lfiltered_hands=filtered_hands;
    }else{//没有检测到手部
        _count++;
        if(_count>_max_predict_frames){
            create();//重新初始化
        }else if(!_lfiltered_hands.empty()){
            for(int i=0;i<_lfiltered_hands.size();i++){
                std::vector<cv::Point3f> current_hand;
                for(int j=0;j<21;j++){
                    cv::KalmanFilter kf=_kfs[i][j];
                    kf.transitionMatrix=(cv::Mat_<float>(9,9)<<
                        1,0,0,dt,0,0,0.5*dt*dt,0,0,
                        0,1,0,0,dt,0,0,0.5*dt*dt,0,
                        0,0,1,0,0,dt,0,0,0.5*dt*dt,
                        0,0,0,1,0,0,dt,0,0,
                        0,0,0,0,1,0,0,dt,0,
                        0,0,0,0,0,1,0,0,dt,
                        0,0,0,0,0,0,1,0,0,
                        0,0,0,0,0,0,0,1,0,
                        0,0,0,0,0,0,0,0,1);
                    kf.predict();
                    cv::Mat state=kf.statePost;
                    current_hand.emplace_back(state.at<float>(0), state.at<float>(1), state.at<float>(2));
                }
                filtered_hands.push_back(current_hand);
            }
            _lfiltered_hands=filtered_hands;
        }
    }
    return filtered_hands;
}

/**
 * @brief 对手势坐标点数据进行卡尔曼滤波,点坐标格式为float
 *        当超过_max_predict_frames帧而没有检测到手势时清零,输出零矩阵
 * @param hands 共享内存中的手势坐标点数据
 * @return 返回滤波后的手势坐标点数据,输出结果可以直接用于opengl-es绘制,非空则有图像,空则无图像
 */
std::vector<float> KalmanFilter::run_gl(const std::vector<float>& glVertices,float dt) 
{
    std::vector<float> filtered_glVertices;
    if(!glVertices.empty()){ //检测到手部 glVertices.size()=63*n
        _count=0;
        for(int idx=0;idx<glVertices.size();idx+=3){
            int hand_idx=idx/63;
            int kp_idx=(idx%63)/3;
            /* 预测阶段 */
            cv::KalmanFilter& kf=_kfs[hand_idx][kp_idx];
            kf.transitionMatrix=(cv::Mat_<float>(9,9)<<
                1,0,0,dt,0,0,0.5*dt*dt,0,0,
                0,1,0,0,dt,0,0,0.5*dt*dt,0,
                0,0,1,0,0,dt,0,0,0.5*dt*dt,
                0,0,0,1,0,0,dt,0,0,
                0,0,0,0,1,0,0,dt,0,
                0,0,0,0,0,1,0,0,dt,
                0,0,0,0,0,0,1,0,0,
                0,0,0,0,0,0,0,1,0,
                0,0,0,0,0,0,0,0,1);
            kf.predict();
            /* 校正阶段 */
            cv::Mat measurement=(cv::Mat_<float>(3,1)<<glVertices[idx],glVertices[idx+1],glVertices[idx+2]);
            kf.correct(measurement);
            /* 获取结果 */
            cv::Mat state=kf.statePost;
            
            filtered_glVertices.push_back(state.at<float>(0));
            filtered_glVertices.push_back(state.at<float>(1));
            filtered_glVertices.push_back(state.at<float>(2));
        }
        _lfiltered_glVertices=filtered_glVertices;
    }else{//没有检测到手部
        _count++;
        if(_count>_max_predict_frames){
            create();//重新初始化
        }else if(!_lfiltered_glVertices.empty()){
            for(int idx=0;idx<_lfiltered_glVertices.size();idx+=3){
                int hand_idx=idx/63;
                int kp_idx=(idx%63)/3;
                /* 预测阶段 */
                cv::KalmanFilter& kf=_kfs[hand_idx][kp_idx];
                kf.transitionMatrix=(cv::Mat_<float>(9,9)<<
                    1,0,0,dt,0,0,0.5*dt*dt,0,0,
                    0,1,0,0,dt,0,0,0.5*dt*dt,0,
                    0,0,1,0,0,dt,0,0,0.5*dt*dt,
                    0,0,0,1,0,0,dt,0,0,
                    0,0,0,0,1,0,0,dt,0,
                    0,0,0,0,0,1,0,0,dt,
                    0,0,0,0,0,0,1,0,0,
                    0,0,0,0,0,0,0,1,0,
                    0,0,0,0,0,0,0,0,1);
                kf.predict();
                /* 获取结果 */
                cv::Mat state=kf.statePost;
                filtered_glVertices.push_back(state.at<float>(0));
                filtered_glVertices.push_back(state.at<float>(1));
                filtered_glVertices.push_back(state.at<float>(2));
            }
        _lfiltered_glVertices=filtered_glVertices;
        }
    }
    return filtered_glVertices;
}


/**
 * @brief 将手势坐标点连接起来
 * @param 输入为滤波后的手势坐标点
 * @return 如果手势坐标点非空,返回非空line_vertices,为则返回空
 *         输出结果可以直接用于opengl-es绘制
 * 
 */
std::vector<float> generate_line_vertices(const std::vector<float>& glVertices) 
{
    std::vector<float> line_vertices;
    for(int i=0;i<glVertices.size()/3;i++){
        int start=hand_connections[i].first;
        int end=hand_connections[i].second;
        std::cout<<"start"<<start<<std::endl;
        std::cout<<"end"<<end<<std::endl;
        line_vertices.insert(line_vertices.end(),{
            glVertices[start],glVertices[start+1],glVertices[start+2],//起始点三维坐标
            glVertices[end],glVertices[end+1],glVertices[end+2]
        });
        
    }
    return line_vertices;
}

