#ifndef CAMERA_H
#define CAMERA_H

#include <EGL/egl.h>
#define GLFW_INCLUDE_ES3    //告诉GLFW使用OpenGL ES
#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

enum Camera_Movement{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

//默认相机参数
const float CAMERA_DEFAULT_YAW=-90.0f;
const float CAMERA_DEFAULT_PITCH=0.0f;
const float CAMERA_DEFAULT_SPEED=2.5f;
//默认鼠标参数
const float MOUSE_DEFAULT_SENSITIVITY=0.1f;
const float MOUSE_DEFAULT_ZOOM=45.0f;
const float MOUSE_MIN_ZOOM=1.0f;
const float MOUSE_MAX_ZOOM=45.0f;
const float MOUSE_MOVE_MODEL_ROTATE=0.2f;
const float MOUSE_MOVE_MODEL_TRANSLATE=0.008f;

struct Mouse
{
    float sensitivity;    /* 鼠标灵敏度 */
    float zoom;           /* 滚轮缩放 */
    bool is_rotating;       /* 旋转模型标志位 */
    bool is_translating;    /* 平移模型标志位 */
    glm::vec2 c_pos;    /* 鼠标当前坐标 */
    glm::vec2 l_pos;    /* 鼠标上一时刻坐标 */
    glm::vec3 model_rotation; // 模型旋转角度(X,Y,Z)

};

struct Timer
{
    float ct;   /* 当前帧时间 */
    float lt;   /* 上一帧时间 */
    float dt;   /* 当前帧与上一帧时间差 */
};


class Camera{
public:
    glm::vec3 position;         /* 摄像机在世界坐标系中的位置 */
    glm::vec3 worldup;          /* 世界坐标系的上向量(默认为(0.0,1.0,0.0)即y轴) */
    glm::vec3 front;            /* 摄像机坐标系的三维向量中的front */
    glm::vec3 up;
    glm::vec3 right;
    float yaw;                  /* 摄像机yaw角 */
    float pitch;    
    float speed;                /* 移动的速度 */
    Mouse mouse;   
    Timer timer;            

    /* 构造函数:用于初始化对象 */
    Camera(glm::vec3 Position=glm::vec3(0.0f));
    glm::mat4 get_view_matrix() const;
    void keyboard_input(Camera_Movement direction,float dt);
    void mouse_move_view(float xoffset,float yoffset);
    void mouse_move_model(float xpos,float ypos,glm::vec3& model_pos);
    void mouse_scroll(float yoffset);

private:
    void update();
};



#endif