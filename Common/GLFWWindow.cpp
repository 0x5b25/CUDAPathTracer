
#include "GLFWWindow.hpp"

#include <cassert>

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

GLFWWindow::~GLFWWindow()
{
    glfwDestroyWindow(handle);
    glfwTerminate();
}

void GLFWWindow::DrawBitmap(unsigned width, unsigned height, const std::uint8_t* data)
{
    //GLuint readFBO = 0;
    //glGenTextures(1, &readFBO);
    //glGenFramebuffers(1, &readFBO);
    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, width, height, 0, GL_RGBA,
        texelType, data);

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)width, 0.f, (float)height, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)height, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)width, (float)height, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)width, 0.f, 0.f);
    }
    glEnd();
}

GLFWWindow::GLFWWindow(const std::string& title, unsigned width, unsigned height)
{
    glfwSetErrorCallback(glfw_error_callback);
    // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);
}

/*! callback for a window resizing event */
static void GLFWWindow_reshape_cb(GLFWwindow* window, int width, int height)
{
    GLFWWindow* gw = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->resize(CUDATracer::Math::vec2i{width, height});
    // assert(GLFWWindow::current);
    //   GLFWWindow::current->resize(vec2i(width,height));
}

/*! callback for a key press */
static void GLFWWindow_key_cb(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    GLFWWindow* gw = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    //if (action == GLFW_PRESS) {
        gw->key(key, action, mods);
    //}
}

/*! callback for _moving_ the mouse to a new position */
static void GLFWWindow_mouseMotion_cb(GLFWwindow* window, double x, double y)
{
    GLFWWindow* gw = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    gw->mouseMotion(x, y);
}

/*! callback for pressing _or_ releasing a mouse button*/
static void GLFWWindow_mouseButton_cb(GLFWwindow* window, int button, int action, int mods)
{
    GLFWWindow* gw = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);
    // double x, y;
    // glfwGetCursorPos(window,&x,&y);
    gw->mouseButton(button, action, mods);
}

void GLFWWindow::run()
{
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i{width, height});

    // glfwSetWindowUserPointer(window, GLFWWindow::current);
    glfwSetFramebufferSizeCallback(handle, GLFWWindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, GLFWWindow_mouseButton_cb);
    glfwSetKeyCallback(handle, GLFWWindow_key_cb);
    glfwSetCursorPosCallback(handle, GLFWWindow_mouseMotion_cb);

    while (!glfwWindowShouldClose(handle)) {
        Update();
        render();

        glfwSwapBuffers(handle);
        glfwPollEvents();
    }
}

