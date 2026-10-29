#include "camera.h"

Camera::Camera(glm::vec3 Position)
{
    position=Position;
    worldup=glm::vec3(0.0f,1.0f,0.0f);
    front=glm::vec3(0.0f,0.0f,-1.0f);// 默认朝向-Z轴
    yaw=CAMERA_DEFAULT_YAW;
    pitch=CAMERA_DEFAULT_PITCH;
    speed=CAMERA_DEFAULT_SPEED;
    mouse.sensitivity=MOUSE_DEFAULT_SENSITIVITY;
    mouse.zoom=MOUSE_DEFAULT_ZOOM;
    update();
}

glm::mat4 Camera::get_view_matrix() const
{
    return glm::lookAt(position,position+front,up);
}

/**
 * 处理键盘事件,用于键盘输入
 */
void Camera::keyboard_input(Camera_Movement direction,float dt)
{
    float velocity=speed*dt;
    if(direction==FORWARD){
        position+=front*velocity;
    }else if(direction==BACKWARD){
        position-=front*velocity;
    }else if(direction==LEFT){
        position-=right*velocity;
    }else if(direction==RIGHT){
        position+=right*velocity;
    }
}

/**
 * 处理鼠标移动事件,用于移动相机视角
 */
void Camera::mouse_move_view(float xoffset,float yoffset)
{
    xoffset*=mouse.sensitivity;
    yoffset*=mouse.sensitivity;
    yaw+=xoffset;
    pitch+=yoffset;

    if(pitch>89.0f){
        pitch=89.0f;
    }
    if(pitch<-89.0f){
        pitch=-89.0f;
    }
    update();
}

/**
 * @brief 处理鼠标拖拽事件,用于拖拽模型
 * @param xpos 鼠标x坐标
 * @param ypos 鼠标y坐标
 * @param model_pos 模型位置
 */
void Camera::mouse_move_model(float xpos,float ypos,glm::vec3& model_pos)
{
    mouse.c_pos=glm::vec2(xpos,ypos);
    glm::vec2 delta=mouse.c_pos-mouse.l_pos;
    if(mouse.is_rotating){
    /**
     * 左键拖动旋转
     * delta.y控制上下翻转(X轴旋转)
     * delta.x控制左右旋转(Y轴旋转)
     */


    float rotx=delta.y*MOUSE_MOVE_MODEL_ROTATE;
    float roty=delta.x*MOUSE_MOVE_MODEL_ROTATE;

    //创建旋转四元数
    glm::quat qx=glm::angleAxis(glm::radians(rotx),right);
    glm::quat qy=glm::angleAxis(glm::radians(roty),up);
    glm::quat q=qy*qx;

    glm::vec3 orbit_center = position + front * 3.0f; // 3.0f是初始距离
        
        // 应用旋转
    model_pos = glm::rotate(q, model_pos - orbit_center) + orbit_center;
    mouse.model_rotation.x+=rotx;
    mouse.model_rotation.y+=roty;


    }else if(mouse.is_translating){
        model_pos+=right*delta.x*MOUSE_MOVE_MODEL_TRANSLATE;
        model_pos-=up*delta.y*MOUSE_MOVE_MODEL_TRANSLATE;
    }
    mouse.l_pos=mouse.c_pos;

}

/**
 * 处理鼠标滚轮事件,用于缩放画面
 */
void Camera::mouse_scroll(float yoffset)
{
    mouse.zoom-=yoffset;
    if(mouse.zoom<1.0f){
        mouse.zoom=1.0f;
    }
    if(mouse.zoom>45.0f){
        mouse.zoom=45.0f;
    }
}

void Camera::update()
{
    glm::vec3 front_temp;
    front_temp.x=cos(glm::radians(yaw))*cos(glm::radians(pitch));
    front_temp.y=sin(glm::radians(pitch));
    front_temp.z=sin(glm::radians(yaw))*cos(glm::radians(pitch));
    front=glm::normalize(front_temp);
    right=glm::normalize(glm::cross(front,worldup));  
    up=glm::normalize(glm::cross(right,front));
}

