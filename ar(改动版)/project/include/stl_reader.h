#ifndef STL_READER_H
#define STL_READER_H

#include <EGL/egl.h>
#define GLFW_INCLUDE_ES3    //告诉GLFW使用OpenGL ES
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>  // 提供 std::find_if
#include <cctype>     // 提供 std::isspace


struct Vertex{
    glm::vec3 position;     //坐标
    glm::vec3 normal;       //法向量
    // glm::vec2 texcoords;    //纹理坐标
    // glm::vec3 tangent;      //切线向量(用于法线贴图)
    // glm::vec3 bitangent;    //副切线向量(用于法线贴图)
     /**
     * @brief 用于比较两个Vertex对象的position和normal是否相同
     * @return 相同返回true,否则返回false
     * 
     */
    bool operator==(const Vertex& other)const{
        const float eps=1e-3f;//误差范围
        return glm::all(glm::lessThan(glm::abs(position-other.position),glm::vec3(eps)))
        &&glm::all(glm::lessThan(glm::abs(normal-other.normal),glm::vec3(eps)));
    }
};


/**
 * 哈希表的两个容器std::unordered_map与std::unordered_set
 * 依赖operator==和std::hash<Vertex>来判断元素是否重复
 * 如果插入两个相同的Vertex
 * std::unordered_set<Vertex>只会保留一个
 * std::unordered_map<Vertex, Value>会映射到同一个Value
 * 
 */
namespace std{
    template<> 
    /* 为Vertex提供哈希函数,使其可作为std::unordered_map的键 */
    struct hash<Vertex>{
        size_t operator()(const Vertex& v)const{
            size_t h1=hash<float>()(v.position[0]);
            size_t h2=hash<float>()(v.position[1]);
            size_t h3=hash<float>()(v.position[2]);
            size_t h4=hash<float>()(v.normal[0]);
            size_t h5=hash<float>()(v.normal[1]);
            size_t h6=hash<float>()(v.normal[2]);
            return h1^(h2<<1)^(h3<<2)^(h4<<3)^(h5<<4)^(h6<<5);
        }
    };
}


class STLReader{
public:

    bool read_stl(const std::string& filename);
    /* 获取获取顶点数组 */
    const std::vector<Vertex>& get_vertices() const;
     /* 获取索引数组 */ 
    const std::vector<uint32_t>& get_indices() const;
    /* 获取顶点数量  */ 
    size_t get_vertex_num() const;
    /* 获取三角形数量 */ 
    size_t get_triangle_num() const;
private:
    std::vector<Vertex> vertices;  //顶点数组(存储所有唯一的顶点数据)
    std::vector<uint32_t> indices; //索引数组(存储顶点索引,即引用_vertices中的位置)
    std::unordered_map<Vertex,uint32_t> vertex_to_index;   //记录每个Vertex对应的索引

    uint32_t add_vertex(const Vertex& vertex);
    bool read_bin_stl(const std::string& filename);
    bool read_ascii_stl(const std::string& filename);

};




#endif