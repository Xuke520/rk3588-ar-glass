#ifndef SHADER_H
#define SHADER_H

#include <EGL/egl.h>
#define GLFW_INCLUDE_ES3    //告诉GLFW使用OpenGL ES
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


class Shader{
public:
    unsigned int id;        /* 着色器程序id */

    Shader(const std::string& vertex_path,const std::string& fragment_path);
    ~Shader();

    /* 该函数不会修改Shader对象的任何成员变量,因此标记为const保证安全性 */
    void use() const;
    void set_int(const std::string &name,int value) const;
    void set_float(const std::string &name,float value) const;
    void set_vec2(const std::string &name,const glm::vec2 &value) const;
    void set_vec2(const std::string &name,float x,float y) const;
    void set_vec3(const std::string &name,const glm::vec3 &value) const;
    void set_vec3(const std::string &name,float x,float y,float z) const;
    void set_vec4(const std::string &name,const glm::vec4 &value) const;
    void set_vec4(const std::string &name,float x,float y,float z,float w) const;
    void set_mat2(const std::string &name,const glm::mat2 &mat) const;
    void set_mat3(const std::string &name,const glm::mat3 &mat) const;
    void set_mat4(const std::string &name,const glm::mat4 &mat) const;

private:
    void check_compile_errors(unsigned int shader,std::string type);

};


#endif