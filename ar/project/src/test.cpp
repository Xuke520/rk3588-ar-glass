#include <Python.h>
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

int main()
{
    /* 初始化PyConfig结构体 */
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    /* 设置 Python 路径 */
    std::wstring python_home = L"/home/elf/miniconda3/envs/ar";
    PyConfig_SetString(&config, &config.home, python_home.c_str());

    // 添加额外的模块搜索路径（如果需要）
    // std::wstring python_path = L"/home/elf/miniconda3/envs/ar/lib/python3.12/site-packages";
    // PyWideStringList_Append(&config.module_search_paths, python_path.c_str());

    Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if(!Py_IsInitialized()){
        std::cerr << "Python initialization failed!" << std::endl;
        return 1;
    }


    std::string script_path="/home/elf/ar/python/main.py"; // 或完整路径如"/path/to/main.py"
    FILE* file=fopen(script_path.c_str(),"r");
    if(!file){
        std::cout<<"Could not open Python script file"<<std::endl;
        return 1;
    }
    PyRun_SimpleFile(file,script_path.c_str());
    fclose(file);


    Py_Finalize();
    return 0;

}