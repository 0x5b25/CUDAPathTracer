#pragma once
#include <iostream>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define xstr(s) str(s)
#define str(s) #s

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&) = delete;\
  classname& operator=(const classname&) = delete


#define CHECK_EQ(val, cond) \
    if(val != cond) __debugbreak();\
    if(val == cond) {} \
    else std::cerr << "Error at " __FUNCTION__ ":" __FILE__ ":" xstr(__LINE__) ": "

#ifndef NDEBUG

#define CHECK(cond) \
    if(cond){}else std::cerr << "Check failed: " xstr(cond)\
    ", at file:" __FILE__ ", line:" xstr(__LINE__)\
    <<std::endl
#else
#define CHECK(cond)
#endif



