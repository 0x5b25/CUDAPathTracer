#pragma once

#include "CUDABuffer.hpp"
#include "CUDATracer.hpp"
#include "GLFWWindow.hpp"
#include "Types.hpp"


class RendererWindow : public GLFWWindow {

protected:

    CUDATracer::Math::vec2d lastMousePos = { -1,-1 };
#define __CLAMP(val, lo, hi) (val < lo)? (lo):((val > hi)? hi : val )
    void UpdateCam(CUDATracer::Camera& cam) {
        cam.pos = camParam.pos;
        cam.fov = camParam.fov;
        camParam.zenith = __CLAMP(
            camParam.zenith, 0.f, (float)M_PI
        );

        //front
        float fy = std::cos(camParam.zenith);
        float d = std::sin(camParam.zenith);
        float fx_ = std::cos(camParam.azimuth);
        float fz_ = std::sin(camParam.azimuth);
        float fx = fx_ * d, fz = fz_ * d;
        CUDATracer::Math::vec3f front(fx, fy, fz);
        //right
        float rx = -fz_;
        float rz = fx_;
        CUDATracer::Math::vec3f right(rx, 0, rz);

        CUDATracer::Math::vec3f up = cross(right, front);
        cam.front = front; cam.right = right;
        cam.up = up;
    }

    //Viewport buffers
    CUDATracer::Math::vec2i bitmapSize;
    std::shared_ptr<CUDATracer::CUDABuffer> bitmap;
    std::shared_ptr<CUDATracer::CUDABuffer> accBuffer;
    //std::shared_ptr<CUDATracer::CUDABuffer> gpuBuffer;
//Rendering infos
    std::shared_ptr<CUDATracer::TypedBuffer<CUDATracer::PathTraceSettings>> renderSettings;
    CUDATracer::ITraceable* renderScene;
    CUDATracer::IPathTracer* renderer;

public:
    //Input handles
    struct CameraParams {
        float fov;
        float azimuth, zenith;
        CUDATracer::Math::vec3f pos;
    } camParam;

    RendererWindow(
        const CUDATracer::Scene& scene,
        const std::string& title = "CUDAPathTracer",
        unsigned width = 800, unsigned height = 480
    )
        : GLFWWindow(title, width, height)
        , renderScene(nullptr)
        , camParam{}
        //,bitmap(200*200*4)
    {
        //
        //delete testRenderer;

        //renderer = CUDATracer::MakeCUDATracerProg();
        renderer = CUDATracer::MakeOptixTracerProg();

        renderScene = renderer->CreateTraceable(scene);

        renderSettings = std::make_shared<CUDATracer::TypedBuffer<CUDATracer::PathTraceSettings>>();

        glfwGetCursorPos(handle, &lastMousePos.x, &lastMousePos.y);

        camParam.pos = { -1,1,0 };
        camParam.fov = M_PI / 3;
        camParam.zenith = M_PI / 2;
        auto& settings = renderSettings->GetMutable<0>();
        UpdateCam(settings.cam);

        settings.viewportHeight = height;
        settings.viewportWidth = width;
        settings.frameID = 0;
        settings.maxDepth = 8;
    }

    ~RendererWindow() override {
        renderSettings = nullptr;
        bitmap = nullptr;
        accBuffer = nullptr;
        delete renderScene;
        delete renderer;
    }

    void ResetFrameID() {
        auto& settings = renderSettings->GetMutable<0>();
        settings.frameID = 0;
    }

    void resize(const vec2i& newSize)override {
        bitmapSize = newSize;

        auto newBufferSize = newSize.x * newSize.y * 4;

        //bitmap.resize(newBufferSize);
        //accBuffer.resize(newBufferSize);

        //if (accBuffer == nullptr || accBuffer->size() < newBufferSize * sizeof(float)) {
        accBuffer = std::make_shared<CUDATracer::CUDABuffer>(newBufferSize * sizeof(float));
        bitmap = std::make_shared<CUDATracer::CUDABuffer>(newBufferSize * sizeof(char));
        //}

        auto& settings = renderSettings->GetMutable<0>();
        settings.viewportHeight = newSize.y;
        settings.viewportWidth = newSize.x;
        ResetFrameID();

    }

    void Update() override
    {
        //Update settings
        HandleKeyInput();
        auto& settings = renderSettings->GetMutable<0>();
        UpdateCam(settings.cam);
    }

    void render() override
    {
        //Perform render
        renderer->Trace(
            *renderScene,
            *renderSettings,
            (float*)(accBuffer->mutable_gpu_data()),
            (char*)(bitmap->mutable_gpu_data())
        );
        //Init settings


        //Download data
        auto tbuffer = (std::uint8_t*)bitmap->cpu_data();

        //Combine with accumulation buffer
        auto& settings = renderSettings->GetMutable<0>();
        auto fid = settings.frameID++;

        DrawBitmap(bitmapSize.x, bitmapSize.y, tbuffer);
    }

    void HandleKeyInput() {
        int fwd = 0, right = 0;
        if (glfwGetKey(handle, GLFW_KEY_W) == GLFW_PRESS)
            fwd += 1;
        if (glfwGetKey(handle, GLFW_KEY_S) == GLFW_PRESS)
            fwd -= 1;
        if (glfwGetKey(handle, GLFW_KEY_D) == GLFW_PRESS)
            right += 1;
        if (glfwGetKey(handle, GLFW_KEY_A) == GLFW_PRESS)
            right -= 1;
        if (fwd != 0 || right != 0) {
            auto& cam = renderSettings->GetMutable<0>().cam;
            camParam.pos += cam.front * fwd * 0.1;
            camParam.pos += cam.right * right * 0.1;
            ResetFrameID();
        }
    }

    void mouseMotion(double px, double py) override {

        auto dx = px - lastMousePos.x;
        auto dy = py - lastMousePos.y;
        lastMousePos = { px, py };

        if (std::abs(dx) < 1e-3 && std::abs(dy) < 1e-3) return;

        int state = glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT);
        if (state == GLFW_PRESS) {
            camParam.azimuth += dx / 400;
            camParam.zenith += dy / 400;
            ResetFrameID();
        }
        /* empty - to be subclassed by user */
    }
};

