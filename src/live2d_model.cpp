#include "live2d_model.hpp"
#include <GL/glew.h>
#include <iostream>
#include <CubismFramework.hpp>
#include <ICubismAllocator.hpp>
#include <Model/CubismUserModel.hpp>
#include <Rendering/OpenGL/CubismRenderer_OpenGLES2.hpp>
#include <Motion/CubismMotion.hpp>
#include <Motion/CubismMotionQueueManager.hpp>
#include <Utils/CubismString.hpp>
#include <Id/CubismIdManager.hpp>
#include <Physics/CubismPhysics.hpp>
#include <Utils/CubismJson.hpp>
#include <fstream>
#include <filesystem>
#include <cstring>

using namespace live2d;
using namespace Live2D::Cubism::Framework;

class CustomAllocator : public Csm::ICubismAllocator {
public:
    void* Allocate(const Csm::csmSizeType size) override {
        std::cout << "Allocating " << size << " bytes" << std::endl;
        void* memory = malloc(size);
        if (memory == nullptr) {
            std::cerr << "Failed to allocate memory" << std::endl;
        }
        else {
            std::memset(memory, 0, size);
        }
        return memory;
    }

    void Deallocate(void* memory) override {
        if (memory) {
            std::cout << "Deallocating memory" << std::endl;
            free(memory);
        }
    }

    void* AllocateAligned(const Csm::csmSizeType size, const Csm::csmUint32 alignment) override {
        std::cout << "Allocating aligned " << size << " bytes with alignment " << alignment << std::endl;
        size_t offset = alignment - 1 + sizeof(void*);
        void* p1 = malloc(size + offset);
        if (p1 == NULL) {
            std::cerr << "Failed to allocate aligned memory" << std::endl;
            return NULL;
        }
        void** p2 = (void**)(((size_t)p1 + offset) & ~(alignment - 1));
        p2[-1] = p1;
        std::memset(p2, 0, size);
        return p2;
    }

    void DeallocateAligned(void* alignedMemory) override {
        if (alignedMemory != NULL) {
            std::cout << "Deallocating aligned memory" << std::endl;
            void* p1 = ((void**)alignedMemory)[-1];
            free(p1);
        }
    }
};

static CustomAllocator allocator;

class CustomUserModel : public Csm::CubismUserModel {
public:
    CustomUserModel() {}
    virtual ~CustomUserModel() {
        if (_model) {
            std::cout << "Deleting model" << std::endl;
            _moc->DeleteModel(_model);
            _model = nullptr;
        }
        if (_moc) {
            std::cout << "Deleting MOC" << std::endl;
            Csm::CubismMoc::Delete(_moc);
            _moc = nullptr;
        }
    }

    bool LoadAsset(const Csm::csmChar* fileName) {
        std::cout << "Loading model: " << fileName << std::endl;

        try {
            // Read JSON file
            std::ifstream ifs(fileName);
            if (!ifs.is_open()) {
                std::cerr << "Failed to open model file: " << fileName << std::endl;
                return false;
            }

            std::string jsonStr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            ifs.close();

            // Parse JSON
            auto json = Utils::CubismJson::Create(
                reinterpret_cast<const Csm::csmByte*>(jsonStr.c_str()),
                static_cast<Csm::csmSizeInt>(jsonStr.length())
            );
            if (json == nullptr) {
                std::cerr << "Failed to parse JSON" << std::endl;
                return false;
            }

            // Get MOC file path
            auto& root = json->GetRoot();
            auto& fileReferences = root[Csm::csmString("FileReferences")];
            if (fileReferences.IsNull()) {
                std::cerr << "Failed to find FileReferences node" << std::endl;
                Utils::CubismJson::Delete(json);
                return false;
            }

            auto& mocFile = fileReferences[Csm::csmString("Moc")];
            if (mocFile.IsNull()) {
                std::cerr << "Failed to find Moc file path" << std::endl;
                Utils::CubismJson::Delete(json);
                return false;
            }

            // Build MOC file path
            std::filesystem::path jsonPath(fileName);
            std::filesystem::path mocPath = jsonPath.parent_path() / mocFile.GetRawString();
            std::string mocFilePath = mocPath.string();

            std::cout << "MOC file path: " << mocFilePath << std::endl;

            // Read MOC file
            std::ifstream mocIfs(mocFilePath, std::ios::binary);
            if (!mocIfs.is_open()) {
                std::cerr << "Failed to open MOC file: " << mocFilePath << std::endl;
                Utils::CubismJson::Delete(json);
                return false;
            }

            mocIfs.seekg(0, std::ios::end);
            size_t size = mocIfs.tellg();
            mocIfs.seekg(0, std::ios::beg);

            std::cout << "Allocating buffer for MOC file: " << size << " bytes" << std::endl;

            // Read MOC file in chunks
            const size_t chunkSize = 4 * 1024; // 4KB chunks
            auto buffer = new Csm::csmByte[size];
            if (buffer == nullptr) {
                std::cerr << "Failed to allocate buffer for MOC file" << std::endl;
                Utils::CubismJson::Delete(json);
                return false;
            }

            std::memset(buffer, 0, size);
            size_t totalBytesRead = 0;
            while (totalBytesRead < size) {
                size_t bytesToRead = std::min(chunkSize, size - totalBytesRead);
                mocIfs.read(reinterpret_cast<char*>(buffer + totalBytesRead), bytesToRead);
                size_t bytesRead = mocIfs.gcount();
                if (bytesRead != bytesToRead) {
                    std::cerr << "Failed to read MOC file: only " << bytesRead << " bytes read" << std::endl;
                    delete[] buffer;
                    Utils::CubismJson::Delete(json);
                    return false;
                }
                totalBytesRead += bytesRead;
            }
            mocIfs.close();

            std::cout << "Successfully read MOC file: " << mocFilePath << " (size: " << size << " bytes)" << std::endl;

            // Check MOC file version
            if (size < 8) {
                std::cerr << "MOC file is too small" << std::endl;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            if (buffer[0] != 'M' || buffer[1] != 'O' || buffer[2] != 'C' || buffer[3] != '3') {
                std::cerr << "Invalid MOC file magic number" << std::endl;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            Csm::csmUint32 version = *reinterpret_cast<Csm::csmUint32*>(buffer + 4);
            std::cout << "MOC file version: " << version << std::endl;

            // Print first 32 bytes of MOC file
            std::cout << "MOC file header: ";
            for (size_t i = 0; i < 32 && i < size; ++i) {
                printf("%02X ", buffer[i]);
            }
            std::cout << std::endl;

            // Load MOC
            std::cout << "Creating MOC..." << std::endl;
            try {
                _moc = Csm::CubismMoc::Create(buffer, static_cast<Csm::csmSizeInt>(size));
                if (_moc == nullptr) {
                    std::cerr << "Failed to create MOC" << std::endl;
                    delete[] buffer;
                    Utils::CubismJson::Delete(json);
                    return false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception while creating MOC: " << e.what() << std::endl;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            std::cout << "Successfully created MOC" << std::endl;

            // Create model instance
            std::cout << "Creating model instance..." << std::endl;
            try {
                _model = _moc->CreateModel();
                if (_model == nullptr) {
                    std::cerr << "Failed to create model instance" << std::endl;
                    Csm::CubismMoc::Delete(_moc);
                    _moc = nullptr;
                    delete[] buffer;
                    Utils::CubismJson::Delete(json);
                    return false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception while creating model instance: " << e.what() << std::endl;
                Csm::CubismMoc::Delete(_moc);
                _moc = nullptr;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            std::cout << "Successfully created model instance" << std::endl;

            // Create renderer
            std::cout << "Creating renderer..." << std::endl;
            try {
                CreateRenderer();
                auto renderer = GetRenderer<Csm::Rendering::CubismRenderer_OpenGLES2>();
                if (renderer == nullptr) {
                    std::cerr << "Failed to create renderer" << std::endl;
                    _moc->DeleteModel(_model);
                    _model = nullptr;
                    Csm::CubismMoc::Delete(_moc);
                    _moc = nullptr;
                    delete[] buffer;
                    Utils::CubismJson::Delete(json);
                    return false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception while creating renderer: " << e.what() << std::endl;
                _moc->DeleteModel(_model);
                _model = nullptr;
                Csm::CubismMoc::Delete(_moc);
                _moc = nullptr;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            std::cout << "Successfully created renderer" << std::endl;

            // Setup default parameters
            std::cout << "Setting up model parameters..." << std::endl;
            try {
                SetupModel();
            }
            catch (const std::exception& e) {
                std::cerr << "Exception while setting up model parameters: " << e.what() << std::endl;
                _moc->DeleteModel(_model);
                _model = nullptr;
                Csm::CubismMoc::Delete(_moc);
                _moc = nullptr;
                delete[] buffer;
                Utils::CubismJson::Delete(json);
                return false;
            }

            delete[] buffer;
            Utils::CubismJson::Delete(json);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception while loading model: " << e.what() << std::endl;
            return false;
        }
    }

protected:
    void SetupModel() {
        // Set default parameters
        if (_model == nullptr) return;

        // Set scale
        GetModelMatrix()->SetWidth(2.0f);
        GetModelMatrix()->SetHeight(2.0f);

        // Set position
        GetModelMatrix()->TranslateX(0.0f);
        GetModelMatrix()->TranslateY(0.0f);
    }
};

Live2DModel::Live2DModel() : is_initialized_(false) {
    initialize_cubism();
}

Live2DModel::~Live2DModel() {
    release_cubism();
}

bool Live2DModel::initialize_cubism() {
    if (is_initialized_) {
        return true;
    }

    // Initialize GLEW
    std::cout << "Initializing GLEW..." << std::endl;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return false;
    }

    // Initialize Cubism SDK
    std::cout << "Initializing Cubism SDK..." << std::endl;
    Csm::CubismFramework::Option option;
    option.LogFunction = nullptr;
    option.LoggingLevel = Csm::CubismFramework::Option::LogLevel_Verbose;
    
    if (!Csm::CubismFramework::StartUp(&allocator, &option)) {
        std::cerr << "Failed to start up Cubism SDK" << std::endl;
        return false;
    }

    Csm::CubismFramework::Initialize();
    is_initialized_ = true;
    std::cout << "Cubism SDK initialized successfully" << std::endl;
    return true;
}

void Live2DModel::release_cubism() {
    if (is_initialized_) {
        std::cout << "Releasing Cubism SDK..." << std::endl;
        if (model_) {
            model_.reset();
        }
        Csm::CubismFramework::Dispose();
        is_initialized_ = false;
        std::cout << "Cubism SDK released" << std::endl;
    }
}

bool Live2DModel::load(const std::string& model_path) {
    if (!is_initialized_) {
        if (!initialize_cubism()) {
            return false;
        }
    }

    try {
        // Create model
        std::cout << "Creating custom model..." << std::endl;
        auto custom_model = std::make_unique<CustomUserModel>();
        if (!custom_model->LoadAsset(model_path.c_str())) {
            std::cerr << "Failed to load model" << std::endl;
            return false;
        }
        model_ = std::move(custom_model);
        
        // Set projection matrix
        std::cout << "Setting up projection matrix..." << std::endl;
        projection_matrix_.Scale(1.0f, 1.0f);
        model_->GetRenderer<Csm::Rendering::CubismRenderer_OpenGLES2>()->SetMvpMatrix(&projection_matrix_);
        
        std::cout << "Model loaded successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

void Live2DModel::update() {
    if (model_) {
        model_->GetModel()->Update();
    }
}

void Live2DModel::draw() {
    if (model_) {
        model_->GetRenderer<Csm::Rendering::CubismRenderer_OpenGLES2>()->DrawModel();
    }
} 