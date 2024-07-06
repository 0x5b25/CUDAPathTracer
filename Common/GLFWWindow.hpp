#pragma once

#include <string>
//#define GLFW_INCLUDE_GLCOREARB
#include "GLFW/glfw3.h"
#include "Math/Vec.hpp"


class GLFWWindow {

    GLuint fbTexture{ 0 };
public:
    using vec2i = CUDATracer::Math::vec2i;

    GLFWWindow(
        const std::string& title,
        unsigned width = 1024, unsigned height = 768
    );
    virtual ~GLFWWindow();

    /*Assume RGBA8 format, origin is bottomleft due to opengl conventions*/
    void DrawBitmap(unsigned width, unsigned height, const std::uint8_t* data);

    const vec2i GetWindowSize() const {
        int width, height;
        glfwGetFramebufferSize(handle, &width, &height);
        return{width, height};
    }

    /*! put pixels on the screen ... */
    virtual void Update()
    { /* empty - to be subclassed by user */
    }

    /*! callback that window got resized */
    virtual void resize(const vec2i& newSize)
    { /* empty - to be subclassed by user */
    }

    virtual void key(int key, int action, int mods)
    {}

    /*! callback that window got resized */
    virtual void mouseMotion(double px, double py)
    {}

    /*! callback that window got resized */
    virtual void mouseButton(int button, int action, int mods)
    {}

    inline vec2i getMousePos() const
    {
        double x, y;
        glfwGetCursorPos(handle, &x, &y);
        return vec2i{ (int)x, (int)y };
    }

    /*! re-render the frame - typically part of draw(), but we keep
      this a separate function so render() can focus on optix
      rendering, and now have to deal with opengl pixel copies
      etc */
    virtual void render()
    { /* empty - to be subclassed by user */
    }

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    /*! the glfw window handle */
    GLFWwindow* handle{ nullptr };

};
