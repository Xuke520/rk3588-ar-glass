#ifndef UTILS_H
#define UTILS_H

#include "stl_reader.h"

struct Texture{
    unsigned int id;    //纹理id
    std::string type;   //纹理类型
    std::string path;   //纹理文件路径
};


/**
 * 网格类
 * @param 顶点
 * @param 索引
 * @param 材质
 * @param VAO
 * @param VBO
 * @param EBO
 * 
 */
class Mesh{
public:
    std::vector<Vertex> vertices;   //顶点数组
    std::vector<uint32_t> indices;  //索引数组
    unsigned int VAO;

    Mesh(std::vector<Vertex> Vertices,std::vector<unsigned int> Indices);
    void draw();
private:
    unsigned int VBO,EBO;
    void setup();
};

/**
 * 模型类
 * 
 */
class Model{
public:
    Model(const std::string& path);
    void draw();
private:
    std::vector<Mesh> meshes;
    STLReader stl_reader;
    bool load_model(const std::string& path);

};






#endif