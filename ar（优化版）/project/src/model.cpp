#include "model.h"


Mesh::Mesh(std::vector<Vertex> Vertices,std::vector<unsigned int> Indices)
{
    this->vertices=vertices;
    this->indices=indices;
    setup();
}

void Mesh::setup()
{
    glGenVertexArrays(1,&VAO);
    glGenBuffers(1,&VBO);
    glGenBuffers(1,&EBO);
    glBindVertexArray(VAO);
    /* 顶点数据 */
    glBindBuffer(GL_ARRAY_BUFFER,VBO);
    glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(Vertex),&vertices[0],GL_STATIC_DRAW);  
    /* 索引数据 */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.size()*sizeof(unsigned int),&indices[0],GL_STATIC_DRAW);
    /* 顶点位置属性 */ 
    glEnableVertexAttribArray(0);   
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)0);
    /* 顶点法线属性 */ 
    glEnableVertexAttribArray(1);   
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,normal));
    /* 顶点纹理属性 */ 
    // glEnableVertexAttribArray(2);	
    // glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,texcoords));
    // glEnableVertexAttribArray(3);	
    // glVertexAttribPointer(3,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,tangent));
    // glEnableVertexAttribArray(4);	
    // glVertexAttribPointer(4,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,bitangent));

    glBindVertexArray(0);
}

void Mesh::draw()
{
    //绘制网格
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES,static_cast<unsigned int>(indices.size()),GL_UNSIGNED_INT,0);
    glBindVertexArray(0);
}


Model::Model(const std::string& path)
{
    if(load_model(path)){
        std::cout << "load model success" << std::endl;
        // 获取STLReader中的顶点和索引数据
        const auto& vertices=stl_reader.get_vertices();
        const auto& indices=stl_reader.get_indices();
        
        // 创建Mesh并添加到meshes
        if(!vertices.empty()&&!indices.empty()){
            meshes.emplace_back(vertices,indices);
            std::cout<<"vertices size="<<vertices.size()<<std::endl; 
            std::cout<<"indices size="<<indices.size()<<std::endl; 
        }else{
            std::cout << "Error: No vertices or indices loaded" << std::endl;
        }
    }else{
        std::cout << "model path err" << std::endl;
    }

}

bool Model::load_model(const std::string& path)
{
    
    return stl_reader.read_stl(path);
 
}

void Model::draw()
{
    std::cout<<"draw"<<std::endl;
    for(unsigned int i=0;i<meshes.size();i++){

        meshes[i].draw();
    }
        
}



