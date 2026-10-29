#include "shader.h"

Shader::Shader(const std::string& vertex_path,const std::string& fragment_path)
{
    id=glCreateProgram();
    std::string v_code,f_code;
    std::ifstream v_file,f_file;
    v_file.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    f_file.exceptions(std::ifstream::failbit|std::ifstream::badbit);

    try{
        v_file.open(vertex_path);
        f_file.open(fragment_path);
        std::stringstream v_stream,f_stream;
        v_stream<<v_file.rdbuf();
        f_stream<<f_file.rdbuf();
        v_file.close();
        f_file.close();
        v_code=v_stream.str();
        f_code=f_stream.str();
    }
    catch(std::ifstream::failure&e){
        std::cout<<"shader file read fail:"<<e.what()<<std::endl;
        return;
    }
    const char* v_shader_code=v_code.c_str();
    const char* f_shader_code=f_code.c_str();

    unsigned int vertex,fragment;
    vertex=glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex,1,&v_shader_code,NULL);
    glCompileShader(vertex);
    check_compile_errors(vertex,"VERTEX");

    fragment=glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment,1,&f_shader_code,NULL);
    glCompileShader(fragment);
    check_compile_errors(fragment,"FRAGMENT");

    id=glCreateProgram();
    glAttachShader(id,vertex);
    glAttachShader(id,fragment);

    glLinkProgram(id);
    check_compile_errors(id,"PROGRAM");

    glDeleteShader(vertex);
    glDeleteShader(fragment);

}

Shader::~Shader()
{
    glDeleteProgram(id);
}

void Shader::use() const
{
    glUseProgram(id);
}

void Shader::set_int(const std::string &name,int value) const
{ 
    glUniform1i(glGetUniformLocation(id,name.c_str()),value); 
}

void Shader::set_float(const std::string &name,float value) const{ 
    glUniform1f(glGetUniformLocation(id,name.c_str()),value); 
}
void Shader::set_vec2(const std::string &name,const glm::vec2 &value) const
{ 
    glUniform2fv(glGetUniformLocation(id,name.c_str()),1,&value[0]); 
}
void Shader::set_vec2(const std::string &name,float x,float y) const
{ 
    glUniform2f(glGetUniformLocation(id,name.c_str()),x,y); 
}
void Shader::set_vec3(const std::string &name,const glm::vec3 &value) const
{ 
    glUniform3fv(glGetUniformLocation(id,name.c_str()),1,&value[0]); 
}
void Shader::set_vec3(const std::string &name,float x,float y,float z) const
{ 
    glUniform3f(glGetUniformLocation(id,name.c_str()),x,y,z); 
}
void Shader::set_vec4(const std::string &name,const glm::vec4 &value) const
{ 
    glUniform4fv(glGetUniformLocation(id,name.c_str()),1,&value[0]); 
}
void Shader::set_vec4(const std::string &name,float x,float y,float z,float w) const
{ 
    glUniform4f(glGetUniformLocation(id,name.c_str()),x,y,z,w); 
}
void Shader::set_mat2(const std::string &name,const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(id,name.c_str()),1,GL_FALSE,&mat[0][0]);
}
void Shader::set_mat3(const std::string &name,const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(id,name.c_str()),1,GL_FALSE,&mat[0][0]);
}
void Shader::set_mat4(const std::string &name,const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(id,name.c_str()),1,GL_FALSE,&mat[0][0]);
}

void Shader::check_compile_errors(unsigned int shader,std::string type)
{
    int success;
    char infoLog[1024];
    if(type!="PROGRAM"){
        glGetShaderiv(shader,GL_COMPILE_STATUS,&success);
        if(!success){
            glGetShaderInfoLog(shader,1024,NULL,infoLog);
            std::cout<<"ERROR::SHADER_COMPILATION_ERROR of type: "<<type<<"\n"<<infoLog<<std::endl;
        }
    }else{
        glGetProgramiv(shader,GL_LINK_STATUS,&success);
        if(!success){
            glGetProgramInfoLog(shader,1024,NULL,infoLog);
            std::cout<<"ERROR::PROGRAM_LINKING_ERROR of type: "<<type<<"\n"<<infoLog<<std::endl;
        }
    }
}

