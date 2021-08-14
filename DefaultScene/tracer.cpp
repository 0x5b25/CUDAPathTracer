
#include<iostream>
#include<fstream>
#include<vector>

#include "GLFW/glfw3.h"
#include "common.hpp"
#include "GLFWWindow.hpp"
#include "RendererWindow.hpp"
#include "CUDATracer.hpp"
#include "Types.hpp"

#ifdef __cplusplus
extern "C"{
#endif

    #include "rgbe.h"

#ifdef __cplusplus
}
#endif

using namespace CUDATracer;




Scene SetupScene() {
    auto arr1 = MakeCube();
    //auto [vert, elem] = MakeCube();
    //auto [vert2, elem2] = MakeCube();
    auto sparr1 = MakeSphere();
    //auto [vertsp, elemsp] = MakeSphere();
    //auto [vertsp2, elemsp2] = MakeSphere();
    auto quadarr1 = MakeQuad();
    //auto [vertq, elemq] = MakeQuad();


    SurfaceMaterial mat{};
        mat.color = { 0.9, 0.7, 0.1 };
        mat.roughness = 0.01;
        mat.metallic = 0.95;
        mat.opacity = 1.f;
        mat.n = 1.5;

    SurfaceMaterial mat2{};
        mat2.color     = {0.5, 0.5, 0.9};
        mat2.roughness = 0.0;
        mat2.metallic  = 0.1;
        mat2.opacity   = 0.2;
        mat2.n         = 2.7;
    
    SurfaceMaterial mat3{};
        mat3.color = { 0.3, 0.95, 0.4 };
        mat3.roughness = 0.00009;
        mat3.metallic = 0.01;
        mat3.opacity = 0.2;
        mat3.n = 1.7;

    SurfaceMaterial mat4{};
        mat4.color = { 0.1, 0.95, 0.84 };
        mat4.roughness = 0.7;
        mat4.metallic = 0.9;
        mat4.opacity = 0.3;
        mat4.n = 1.5;

    SurfaceMaterial groundmat{};
        groundmat.color = { 0.9, 0.9, 0.8 };
        groundmat.roughness = 0.5;
        groundmat.metallic = 0.01;
        groundmat.opacity = 1.f;
        groundmat.n = 2.7;

    auto cube2tran = Math::MatMul(Math::MakeTranslate(Math::vec3f(1.3, 0, 1.7))
        ,Math::MatMul(Math::MakeRotation(Math::Quaternion3f::rotate(Math::vec3f(0, 1, 0),0.2f))
        , Math::MakeScale(Math::vec3f(1.2, 1.2, 1.2)))
        );
    
    auto cube1tran = Math::MakeTranslate(Math::vec3f(0, 0.1, -0.5))
        //* glm::toMat4(glm::angleAxis(0.2f, glm::vec3(0, 1, 0)))
        //* glm::scale(glm::vec3(1.2, 1.2, 1.2));
        ;

    auto sphere1tran = Math::MatMul(Math::MakeTranslate(Math::vec3f(-1.5, 0.5, 1.2))
        //* glm::toMat4(glm::angleAxis(0.2f, glm::vec3(0, 1, 0)))
        , Math::MakeScale(Math::vec3f{ .5f, .5f, .5f })
    );

    auto sphere2tran = Math::MatMul(Math::MakeTranslate(Math::vec3f(-1.2, 1, -1.2))
        //* glm::toMat4(glm::angleAxis(0.2f, glm::vec3(0, 1, 0)))
        , Math::MakeScale(Math::vec3f{ .8f, .8f, .8f })
    );
    
    auto quad1tran = Math::MatMul(Math::MakeTranslate(Math::vec3f(-4, 0, -4))
        //* glm::toMat4(glm::angleAxis(0.2f, glm::vec3(0, 1, 0)))
        , Math::MakeScale(Math::vec3f{ 8.f, 1.f, 8.f })
    );


    Mesh cube1 = ConstructMesh(arr1, mat, cube1tran);
    Mesh cube2 = ConstructMesh(arr1, mat2, cube2tran);
    Mesh sphere1 = ConstructMesh(sparr1, mat3, sphere1tran);
    Mesh sphere2 = ConstructMesh(sparr1, mat4, sphere2tran);
    Mesh quad1 = ConstructMesh(quadarr1, groundmat, quad1tran);

    Scene scn{
        /*SkyLight*/{
        /*Math::vec3f color;*/{0.98, 0.97,0.99},
        /*Math::vec3f dir;  */-Math::normalize(Math::vec3f{1,1,1}),
        /*float intensity;  */5
        },
        //{std::move(cube1),std::move(cube2),std::move(sphere1),std::move(quad1)}
        {}
    };
    scn.objects.push_back(std::move(cube1));
    scn.objects.push_back(std::move(cube2));
    scn.objects.push_back(std::move(quad1));
    scn.objects.push_back(std::move(sphere1));
    scn.objects.push_back(std::move(sphere2));
      

    return scn;
}

int main(){

    //Load HDRI envmap
    int mapW, mapH;

    auto f = fopen("./sky.hdr", "rb");
    RGBE_ReadHeader(f, &mapW, &mapH, NULL);
    std::vector<float>image(3 * mapW * mapH);
    RGBE_ReadPixels_RLE(f, image.data(), mapW, mapH);
    fclose(f);

    auto scene = SetupScene();
    scene.envMap = std::move(image);
    scene.envMapDim = Math::vec2ui{(unsigned)mapW, (unsigned)mapH};

    RendererWindow mainWnd{scene};

    mainWnd.run();

}
