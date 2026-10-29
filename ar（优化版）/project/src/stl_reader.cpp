#include "stl_reader.h"

bool STLReader::read_stl(const std::string& filename)
{
    /* 先尝试以二进制文件读取 */
    if(read_bin_stl(filename)){
        return true;
    }else{/* 尝试以ASCII文件读取 */
        return read_ascii_stl(filename);
    }
}

/* 获取获取顶点数组 */
const std::vector<Vertex>& STLReader::get_vertices() const
{
    return vertices;
}

 /* 获取索引数组 */ 
const std::vector<uint32_t>& STLReader::get_indices() const
{
    return indices;
}

/* 获取顶点数量  */ 
size_t STLReader::get_vertex_num() const

{
    return vertices.size();
}
/* 获取三角形数量 */
size_t STLReader::get_triangle_num() const
{
    return indices.size()/3;
}

uint32_t STLReader::add_vertex(const Vertex& vertex)
{
    auto it=vertex_to_index.find(vertex);
    if(it!=vertex_to_index.end()){
        return it->second; //顶点已存在，返回已有索引
    }
    
    // 新顶点，添加到数组
    uint32_t index=static_cast<uint32_t>(vertices.size());
    vertices.push_back(vertex);
    vertex_to_index[vertex]=index;
    return index;
}

bool STLReader::read_bin_stl(const std::string& filename)
{
    std::ifstream file(filename,std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    /* 前80字节为头部信息,跳过 */
    file.seekg(80);
    /* 读取面片数量(4字节小端序) */ 
    uint32_t triangle_num=0;
    file.read(reinterpret_cast<char*>(&triangle_num),sizeof(triangle_num));
    /* 检查文件大小是否匹配 */
    file.seekg(0,std::ios::end);
    size_t file_size=file.tellg();
    if(file_size!=84+triangle_num*50){
        file.close();
        return false;  
    }
    file.seekg(84);  // 回到数据开始位置

    vertices.reserve(triangle_num*3);
    indices.reserve(triangle_num*3);

    // 读取所有三角形数据
    for(uint32_t i=0;i<triangle_num;++i){
        Vertex v1,v2,v3;

        file.read(reinterpret_cast<char*>(&v1.normal),12);  // 法线
        file.read(reinterpret_cast<char*>(&v1.position),12);     // 顶点1
        file.read(reinterpret_cast<char*>(&v2.position),12);     // 顶点2
        file.read(reinterpret_cast<char*>(&v3.position),12);     // 顶点3

        // 三个顶点共享同一个法线
        v2.normal=v1.normal;
        v3.normal=v1.normal;
    
        // 添加顶点并获取索引
        uint32_t i1=add_vertex(v1);
        uint32_t i2=add_vertex(v2);
        uint32_t i3=add_vertex(v3);

        // 添加三角形索引
        indices.push_back(i1);
        indices.push_back(i2);
        indices.push_back(i3);

        file.seekg(2,std::ios::cur);  // 跳过2字节属性
    }
    file.close();
    return true;
}

// 读取ASCII STL文件
bool STLReader::read_ascii_stl(const std::string& filename)
{
    std::ifstream file(filename);
    if(!file.is_open()){
        return false;
    }

    std::string line;
    Vertex current_vertex;
    bool inFacet=false;
    int vertex_num=0;

    while(std::getline(file,line)){
        // 去除行首尾空白字符
        line.erase(line.begin(),std::find_if(line.begin(),line.end(),[](int ch){return !std::isspace(ch);}));
        line.erase(std::find_if(line.rbegin(),line.rend(),[](int ch){return !std::isspace(ch);}).base(),line.end());

        if(line.empty()) continue;

        if(line.find("facet normal")==0){
            // 读取法线
            sscanf(line.c_str(),"facet normal %f %f %f",&current_vertex.normal[0],&current_vertex.normal[1],&current_vertex.normal[2]);
            inFacet=true;
            vertex_num=0;
        }else if(line.find("vertex")==0&&inFacet){
            // 读取顶点
            sscanf(line.c_str(),"vertex %f %f %f",&current_vertex.position[0],&current_vertex.position[1],&current_vertex.position[2]);
            uint32_t index=add_vertex(current_vertex);
            indices.push_back(index);
            vertex_num++;
        }else if(line.find("endfacet")==0){
            inFacet=false;
            if(vertex_num!=3){
                file.close();
                vertices.clear();
                indices.clear();
                return false;
            }
        }
    }
    file.close();
    return !vertices.empty();  // 如果读取到三角形则认为成功
}



