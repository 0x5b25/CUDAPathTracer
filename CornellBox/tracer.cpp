
#include<iostream>
#include<fstream>
#include<vector>

#include "GLFW/glfw3.h"
#include "common.hpp"
#include "GLFWWindow.hpp"
#include "RendererWindow.hpp"
#include "CUDATracer.hpp"
#include "Types.hpp"

using namespace CUDATracer;


Scene SetupScene() {

    SurfaceMaterial diffuseMat{};
        diffuseMat.color = 1.0f;
        diffuseMat.roughness = 0.5f;
        diffuseMat.metallic = 0.0f;
        diffuseMat.opacity = 1.0f;
        diffuseMat.n = 2.5f; //approx. for cement

    auto floor = ConstructMesh(MakeQuad(),diffuseMat,
        Math::MatMul(
            Math::MakeTranslate(Math::vec3f{-2,-2,-2}),
            Math::MakeScale(Math::vec3f{4,1,4})
        )
    );
    auto ceilling = ConstructMesh(MakeQuad(), diffuseMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 0,0,1 }, M_PI)),
            Math::MatMul(
            Math::MakeTranslate(Math::vec3f{ -2,-2,-2 }),
            Math::MakeScale(Math::vec3f{ 4,1,4 })
            )
        )
    );

    auto back = ConstructMesh(MakeQuad(), diffuseMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 0,0,1 }, M_PI/2)),
            Math::MatMul(
            Math::MakeTranslate(Math::vec3f{ -2,-2,-2 }),
            Math::MakeScale(Math::vec3f{ 4,1,4 })
            )
        )
    );

    auto front = ConstructMesh(MakeQuad(), diffuseMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 0,0,1 }, -M_PI / 2)),
            Math::MatMul(
                Math::MakeTranslate(Math::vec3f{ -2,-2,-2 }),
                Math::MakeScale(Math::vec3f{ 4,1,4 })
            )
        )
    );

    SurfaceMaterial leftMat = diffuseMat;
    leftMat.color = {1,0,0};
    auto left = ConstructMesh(MakeQuad(), leftMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 1,0,0 }, M_PI / 2)),
            Math::MatMul(
                Math::MakeTranslate(Math::vec3f{ -2,-2,-2 }),
                Math::MakeScale(Math::vec3f{ 4,1,4 })
            )
        )
    );

    SurfaceMaterial rightMat = diffuseMat;
    rightMat.color = { 0,1,0 };
    auto right = ConstructMesh(MakeQuad(), rightMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 1,0,0 }, -M_PI / 2)),
            Math::MatMul(
                Math::MakeTranslate(Math::vec3f{ -2,-2,-2 }),
                Math::MakeScale(Math::vec3f{ 4,1,4 })
            )
        )
    );


    SurfaceMaterial lightMat = diffuseMat;
    lightMat.emissiveColor = {1.f,.8f, 0.5f};
    lightMat.emissiveStrength = 10;

    auto light = ConstructMesh(MakeQuad(), lightMat,
        Math::MatMul(
            Math::MakeRotation(Math::Quaternion3f::rotate({ 0,0,1 }, M_PI)),
            Math::MakeTranslate(Math::vec3f{ -0.5,-1.99999f,-0.5 })
        )
    );

    SurfaceMaterial boxMat = diffuseMat;
    boxMat.roughness = 0;
    boxMat.metallic = 1;
    //boxMat.color = { 0,1,0 };
    auto box1 = ConstructMesh(MakeCube(), boxMat,
        Math::MatMul(
            Math::MakeTranslate(Math::vec3f{ 0.5 - 0.5,-2,-0.5 - 0.5 }),
            Math::MatMul(
                Math::MakeRotation(Math::Quaternion3f::rotate({ 0,1,0 }, 0.4)),
                Math::MakeScale(Math::vec3f{ 1,2.5,1 })
            )
        )
    );


    SurfaceMaterial boxMat2 = diffuseMat;
    boxMat2.roughness = 0.5;
    boxMat2.metallic = 0;
    //boxMat.color = { 0,1,0 };
    auto box2 = ConstructMesh(MakeCube(), boxMat2,
        Math::MatMul(
            Math::MakeTranslate(Math::vec3f{ -0.5 - 0.5,-2,0.5 - 0.5 }),
            Math::MakeRotation(Math::Quaternion3f::rotate({ 0,1,0 }, -0.4))
        )
    );

    Scene scn{
        /*SkyLight*/{
        /*Math::vec3f color;*/{0.98, 0.97,0.99},
        /*Math::vec3f dir;  */-Math::normalize(Math::vec3f{1,1,1}),
        /*float intensity;  */0
        },
        //{std::move(cube1),std::move(cube2),std::move(sphere1),std::move(quad1)}
        {}
    };
    scn.objects.push_back(std::move(floor));
    scn.objects.push_back(std::move(ceilling));
    scn.objects.push_back(std::move(back));
    //scn.objects.push_back(std::move(front));
    scn.objects.push_back(std::move(left));
    scn.objects.push_back(std::move(right));
    scn.objects.push_back(std::move(light));
    scn.objects.push_back(std::move(box1));
    scn.objects.push_back(std::move(box2));

    return scn;
}

class AppWindow : public RendererWindow
{
    public:
    AppWindow()
        :RendererWindow(SetupScene())
    {
        auto& settings = renderSettings->GetMutable<0>();
        settings.maxDepth = 4;

        camParam.pos = { -6,0,0 };
        camParam.fov = M_PI / 3;
        camParam.zenith = M_PI / 2;
        UpdateCam(settings.cam);
    }

    //void render() override
    //{
    //    //Perform render
    //    renderer->Trace(
    //        renderScene,
    //        renderSettings,
    //        (float*)(accBuffer->mutable_gpu_data()),
    //        (char*)(bitmap->mutable_gpu_data())
    //    );
    //    //Init settings
    //
    //
    //    //Download data
    //    auto tbuffer = (std::uint8_t*)bitmap->cpu_data();
    //
    //    //Combine with accumulation buffer
    //    auto& settings = renderSettings.GetMutable<0>();
    //    auto fid = settings.frameID++;
    //
    //    DrawBitmap(bitmapSize.x, bitmapSize.y, tbuffer);
    //}
};

int main(){

    //Load HDRI envmap
    int mapW, mapH;
        
    AppWindow mainWnd{};

    mainWnd.run();

}
