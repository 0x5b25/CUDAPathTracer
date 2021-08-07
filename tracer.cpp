
#include<iostream>
#include<fstream>
#include<vector>

#include "GLFW/glfw3.h"
#include "Common/common.hpp"
#include "Common/GLFWWindow.hpp"
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


inline std::tuple<
    std::vector<Vertex>,
    std::vector<std::uint32_t>
> MakeCube(const Math::vec3f& offset = {}) {
#define NX(v) PX(0,0,v)
#define NY(v) PY(0,0,v)
#define NZ(v) PZ(0,0,v)
#define PX(a,b,x) {x,a,b}
#define PY(a,b,x) {a,x,b}
#define PZ(a,b,x) {a,b,x}
#define QUAD(AX, axval, norm)\
{AX(0,0,axval),norm,{0,0}},{AX(1,0,axval),norm,{1,0}},\
{AX(1,1,axval),norm,{1,1}},{AX(0,1,axval),norm,{0,1}}

    std::vector<Vertex> v{
        /*bottom*/QUAD(PY, 0, NY(-1)),/*top*/QUAD(PY, 1, NY(1)),
        /*x+*/QUAD(PX,1,NX(1)),/*x-*/QUAD(PX,0,NX(-1)),
        /*z+*/QUAD(PZ, 1, NZ(1)),/*z-*/QUAD(PZ, 0, NZ(-1))
    };
#define wind(a,b,c,d) a,b,c,a,c,d
    std::vector<std::uint32_t> e{
        /*top*/    wind(0,1,2,3),
        /*bottom*/ wind(7,6,5,4),
        /*front*/  wind(8,9,10,11),
        /*back*/   wind(15,14,13,12),
        /*left*/   wind(16,17,18,19),
        /*right*/  wind(23,22,21,20)
    };
#undef wind

    for (auto& vt : v) {
        vt.position += offset;
    }

    return std::make_tuple(v, e);
}


std::tuple<
    std::vector<Vertex>,
    std::vector<std::uint32_t>
> MakeSphere() {

    unsigned seg_y = 40;
    unsigned seg_x = 80;

    std::vector<Vertex> v{
        {{0,1,0},{0,1,0},{0.5,1}},//top
        {{0,-1,0},{0,-1,0},{0.5,0}},//bottom
    };
    std::vector<std::uint32_t> e{};

    float delta_x = M_PI * 2 / seg_x;
    float delta_y = M_PI / seg_y;

    //low to high
    for (int y = 1; y < seg_y; y++) {
        float zenith = y * delta_y;
        float r = std::sin(zenith);
        float height = -std::cos(zenith);
        float uv_y = y / seg_y;
        for (int x = 0; x < seg_x; x++) {
            float azimuth = x * delta_x;
            float coord_x = std::cos(azimuth) * r;
            float coord_z = std::sin(azimuth) * r;
            float uv_x = x / seg_x;
            v.push_back({
                {coord_x, height, coord_z},
                {coord_x, height, coord_z},
                {uv_x, uv_y}
                });
        }
    }
#define wind(a,b,c,d) a,b,c,a,c,d

    //link elements
    for (int y = 0; y < seg_y - 2; y++) {
        int next_y = y + 1;
        for (int x = 0; x < seg_x; x++) {
            int next_x = (x + 1) % seg_x;

            unsigned id_l1 = 2 + y * seg_x + x;
            unsigned id_t1 = 2 + next_y * seg_x + x;

            unsigned id_l2 = 2 + y * seg_x + next_x;
            unsigned id_t2 = 2 + next_y * seg_x + next_x;

            e.insert(e.end(), { wind(id_l2, id_l1, id_t1, id_t2) });

        }
    }

#undef wind
    //seal top and bottom
    unsigned idx_top = v.size() - seg_x;
    unsigned idx_bottom = 2;
    for (int x = 0; x < seg_x; x++) {
        int next_x = (x + 1) % seg_x;
        e.insert(e.end(), { 0, idx_top + next_x, idx_top + x, 1, idx_bottom + x, idx_bottom + next_x });
    }


    return std::make_tuple(v, e);
}

std::tuple<
    std::vector<Vertex>,
    std::vector<std::uint32_t>
> MakeQuad(Math::vec2f size = { 1,1 }, Math::vec3f offset = {}) {
    std::vector<Vertex> v{
        //lower
        {Math::vec3f{0,0,0} + offset,{0,1,0},{0,0}},
        {Math::vec3f{0,0,size.x} + offset,{0,1,0},{0,1}},
        {Math::vec3f{size.y,0,size.x} + offset,{0,1,0},{1,1}},
        {Math::vec3f{size.y,0,0} + offset,{0,1,0},{1,0}},
    };
#define wind(a,b,c,d) a,b,c,a,c,d
    std::vector<std::uint32_t> e{
        /*top*/    wind(0,1,2,3),
    };
#undef wind
    return std::make_tuple(v, e);
}

Mesh ConstructMesh(const std::tuple<
    std::vector<Vertex>,
    std::vector<std::uint32_t>
>& pair,const SurfaceMaterial& mat, const Math::mat4x4f& transform) {
    auto vert = std::get<0>(pair);
    auto& indi = std::get<1>(pair);

    for (auto& v : vert) {
        v.position = Math::MatMul(transform, v.position, 1.f);
        v.normal = Math::MatMul(transform, v.normal, 0.f);
    }

    return Mesh(vert, indi, mat);

}

Scene SetupScene() {
    auto arr1 = MakeCube();
    //auto [vert, elem] = MakeCube();
    //auto [vert2, elem2] = MakeCube();
    auto sparr1 = MakeSphere();
    //auto [vertsp, elemsp] = MakeSphere();
    //auto [vertsp2, elemsp2] = MakeSphere();
    auto quadarr1 = MakeQuad();
    //auto [vertq, elemq] = MakeQuad();


    SurfaceMaterial mat{
        /*glm::vec3 color;*/{0.9, 0.7, 0.1},
        /*float roughness;*/0.01,
        /*float metallic; */0.95,
        /*float opacity;  */1.f,
        /*float refract*/    1.5,
    };
    SurfaceMaterial mat2{
        /*glm::vec3 color;*/{0.5, 0.5, 0.9},
        /*float roughness;*/0.0,
        /*float metallic; */0.1,
        /*float opacity;  */0.2,
        /*float refract*/    2.7,
    };
    SurfaceMaterial mat3{
        /*glm::vec3 color;*/{0.3, 0.95, 0.4},
        /*float roughness;*/0.00009,
        /*float metallic; */0.01,
        /*float opacity;  */0.2,
        /*float refract*/    1.7,
    };

    SurfaceMaterial mat4{
        /*glm::vec3 color;*/{0.1, 0.95, 0.84},
        /*float roughness;*/0.7,
        /*float metallic; */0.9,
        /*float opacity;  */0.3,
        /*float refract*/   1.5,
    };

    SurfaceMaterial groundmat{
        /*glm::vec3 color;*/{0.9, 0.9, 0.8},
        /*float roughness;*/0.5,
        /*float metallic; */0.01,
        /*float opacity;  */1.f,
        /*float refract*/   2.7,
    };

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

class RendererWindow : public GLFWWindow {

//Input handles
    struct CameraParams {
        float fov;
        float azimuth, zenith;
        Math::vec3f pos;
    } camParam;

    Math::vec2d lastMousePos = { -1,-1 };
#define __CLAMP(val, lo, hi) (val < lo)? (lo):((val > hi)? hi : val )
    void UpdateCam(Camera& cam) {
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
        Math::vec3f front(fx, fy, fz);
        //right
        float rx = -fz_;
        float rz = fx_;
        Math::vec3f right(rx,0,rz);

        Math::vec3f up = Math::cross(right, front);
        cam.front = front; cam.right = right;
        cam.up = up;
    }

//Viewport buffers
    vec2i bitmapSize;
    std::shared_ptr<CUDATracer::CUDABuffer> bitmap;
    std::shared_ptr<CUDATracer::CUDABuffer> accBuffer;
    //std::shared_ptr<CUDATracer::CUDABuffer> gpuBuffer;
//Rendering infos
    TypedBuffer<PathTraceSettings> renderSettings;
    CUDAScene renderScene;
    PathTracer renderer;

public:
    RendererWindow(const Scene& scene, unsigned width = 800, unsigned height = 480)
        : GLFWWindow("CUDAPathTracer", width, height)
        , camParam{}
        , renderScene(scene)
        //,bitmap(200*200*4)
    {
        glfwGetCursorPos(handle, &lastMousePos.x, &lastMousePos.y);

        camParam.pos = { -2,1,0 };
        camParam.fov = M_PI / 3;      
        camParam.zenith = M_PI / 2;
        auto& settings = renderSettings.GetMutable<0>();
        UpdateCam(settings.cam);

        settings.viewportHeight = height;
        settings.viewportWidth = width;
        settings.frameID = 0;
        settings.maxDepth = 6;
    }

    void ResetFrameID() {
        auto& settings = renderSettings.GetMutable<0>();
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

        auto& settings = renderSettings.GetMutable<0>();
        settings.viewportHeight = newSize.y;
        settings.viewportWidth = newSize.x;
        ResetFrameID();

    }

    void draw() override {
        //Update settings
        HandleKeyInput();
        auto& settings = renderSettings.GetMutable<0>();
        UpdateCam(settings.cam);

        //Perform render
        renderer.Trace(
            renderScene, 
            renderSettings, 
            (float*)(accBuffer->mutable_gpu_data()),
            (char*)(bitmap->mutable_gpu_data())
        );
        //Init settings
        

        //Download data
        auto tbuffer = (std::uint8_t*)bitmap->cpu_data();

        //Combine with accumulation buffer
        settings = renderSettings.GetMutable<0>();
        auto fid = settings.frameID++;
       
        DrawBitmap(bitmapSize.x,bitmapSize.y, tbuffer);
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
            auto& cam = renderSettings.GetMutable<0>().cam;
            camParam.pos += cam.front * fwd * 0.1;
            camParam.pos += cam.right * right * 0.1;
            ResetFrameID();
        }
    }

    void mouseMotion(double px, double py) override {
        
        auto dx = px - lastMousePos.x;
        auto dy = py - lastMousePos.y;
        lastMousePos = {px, py};

        if(std::abs(dx) < 1e-3 && std::abs(dy) < 1e-3) return;

        int state = glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_LEFT);
        if (state == GLFW_PRESS) {
            camParam.azimuth += dx / 400;
            camParam.zenith += dy / 400;
            ResetFrameID();
        }
        /* empty - to be subclassed by user */
    }
      


};


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

    //std::cout << "Hello World!" << std::endl;
    //
    //GLFWwindow* window;
    //
    ///* Initialize the library */
    //if (!glfwInit())
    //    return -1;
    //
    ///* Create a windowed mode window and its OpenGL context */
    //window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    //if (!window)
    //{
    //    glfwTerminate();
    //    return -1;
    //}
    //
    ///* Make the window's context current */
    //glfwMakeContextCurrent(window);
    //
    ///* Loop until the user closes the window */
    //while (!glfwWindowShouldClose(window))
    //{
    //    /* Render here */
    //    glClear(GL_COLOR_BUFFER_BIT);
    //
    //    /* Swap front and back buffers */
    //    glfwSwapBuffers(window);
    //
    //    /* Poll for and process events */
    //    glfwPollEvents();
    //}
    //
    //glfwTerminate();
    //return 0;
}
