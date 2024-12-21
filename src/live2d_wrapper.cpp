#include "../include/live2d_wrapper.hpp"
#include <GL/gl.h>
#include <iostream>

Live2DModel::Live2DModel()
    : model_(nullptr)
    , viewMatrix_(nullptr)
    , modelMatrix_(nullptr)
    , isInitialized_(false)
{
}

Live2DModel::~Live2DModel() {
    if (model_) {
        model_.reset();
    }
    if (viewMatrix_) {
        delete viewMatrix_;
    }
    if (modelMatrix_) {
        delete modelMatrix_;
    }
    CubismFramework::Dispose();
}

bool Live2DModel::initialize() {
    // 初始化Cubism Framework
    CubismFramework::Option option;
    option.LogFunction = nullptr;
    option.LoggingLevel = CubismFramework::Option::LogLevel_Verbose;
    
    if (CubismFramework::StartUp(&option)) {
        if (CubismFramework::Initialize()) {
            std::cout << "Cubism Framework 初始化成功" << std::endl;
            isInitialized_ = true;
            return true;
        }
    }
    
    std::cout << "Cubism Framework 初始化失败" << std::endl;
    return false;
}

bool Live2DModel::loadModel(const std::string& modelPath) {
    if (!isInitialized_) {
        std::cout << "Framework未初始化" << std::endl;
        return false;
    }
    
    try {
        // 创建模型实例
        model_ = std::make_unique<CubismUserModel>();
        
        // 加载模型数据
        // TODO: 实现模型加载逻辑
        
        // 初始化矩阵
        viewMatrix_ = new CubismMatrix44();
        modelMatrix_ = new CubismMatrix44();
        
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "模型加载失败: " << e.what() << std::endl;
        return false;
    }
}

void Live2DModel::update() {
    if (!model_) return;
    
    // 更新模型
    model_->Update();
}

void Live2DModel::draw() {
    if (!model_) return;
    
    // 获取OpenGL渲染器
    CubismRenderer_OpenGL* renderer = 
        static_cast<CubismRenderer_OpenGL*>(model_->GetRenderer());
    
    // 设置投影矩阵
    renderer->SetMvpMatrix(viewMatrix_);
    
    // 渲染模型
    model_->Draw();
}

void Live2DModel::setParameter(const std::string& paramId, float value) {
    if (!model_) return;
    
    // 设置参数值
    // TODO: 实现参数设置
}

float Live2DModel::getParameter(const std::string& paramId) {
    if (!model_) return 0.0f;
    
    // 获取参数值
    // TODO: 实现参数获取
    return 0.0f;
}

void Live2DModel::setViewMatrix(float* matrix44) {
    if (!viewMatrix_) return;
    viewMatrix_->SetMatrix(matrix44);
}

void Live2DModel::setModelMatrix(float* matrix44) {
    if (!modelMatrix_) return;
    modelMatrix_->SetMatrix(matrix44);
}

PYBIND11_MODULE(live2d_core, m) {
    py::class_<Live2DModel>(m, "Live2DModel")
        .def(py::init<>())
        .def("initialize", &Live2DModel::initialize)
        .def("load_model", &Live2DModel::loadModel)
        .def("update", &Live2DModel::update)
        .def("draw", &Live2DModel::draw)
        .def("set_parameter", &Live2DModel::setParameter)
        .def("get_parameter", &Live2DModel::getParameter)
        .def("set_view_matrix", &Live2DModel::setViewMatrix)
        .def("set_model_matrix", &Live2DModel::setModelMatrix);
} 