#include "shader.h"
#include "camera.h"
#include "invoke_python.h"
#define STB_IMAGE_IMPLEMENTATION    
#include "stb_image.h"
#include <GLES2/gl2.h>

const unsigned int WINDOW_WIDTH=1920;
const unsigned int WINDOW_HEIGHT=1080;
#define WINDOW_NAME    "learn window"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main()
{

    if(python_init("/home/elf/miniconda3/envs/ar","/home/elf/ar/python/main.py")==false){
        std::cout<<"python init failed"<<std::endl;
    }


    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
    glfwWindowHint(GLFW_CLIENT_API,GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API,GLFW_EGL_CONTEXT_API);
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH,WINDOW_HEIGHT,WINDOW_NAME,NULL,NULL);
    if(NULL==window){
        std::cout<<"Failed to create GLFW window"<<std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window,framebuffer_size_callback);
    glEnable(GL_DEPTH_TEST);

    Shader shader("./../shader-src/vs.src","./../shader-src/fs.src");
    SharedMemory reader("/gesture");
    std::vector<float> glVertices;
    KalmanFilter kf;

    unsigned int VBO,VAO;
    glGenVertexArrays(1,&VAO);
    glGenBuffers(1,&VBO);

    while(!glfwWindowShouldClose(window)){
        
        glVertices.clear();
        reader.read_gl_landmarks(glVertices,2.0f);
        std::vector<float> out=kf.run_gl(glVertices,1/100);
        std::vector<float> out1=generate_line_vertices(out); 

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER,out1.size()*sizeof(float),out1.data(),GL_DYNAMIC_DRAW);
        /* 顶点属性的位置 顶点属性的分量数量 数据类型 是否标准化 步长 偏移量 */
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);


        glClearColor(0.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);


        processInput(window);
        shader.use();

        glm::mat4 model=glm::mat4(1.0f);
        glm::mat4 view=glm::mat4(1.0f);
        view=glm::lookAt((glm::vec3){0.0f,0.0f,3.0f},(glm::vec3){0.0f,0.0f,0.0f},(glm::vec3){0.0f,1.0f,0.0f});
        glm::mat4 projection=glm::perspective(glm::radians(45.0f),(float)WINDOW_WIDTH/WINDOW_HEIGHT,0.1f,100.0f);
        shader.set_mat4("view",view);
        shader.set_mat4("model",model);
        shader.set_mat4("projection",projection);


        glBindVertexArray(VAO);
        glDrawArrays(GL_LINES,0,out1.size()/3);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1,&VAO);
    glDeleteBuffers(1,&VBO);

    glfwTerminate();
    return 0;

}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window,true);
}


void framebuffer_size_callback(GLFWwindow* window,int width,int height)
{
    glViewport(0,0,width,height);
}




