#include "hello.h"
#include <sstream>

std::string hello_world(const std::string& name) {
    std::ostringstream oss;
    oss << "Hello, " << name << "! Welcome to pybind11!";
    return oss.str();
}

int add_numbers(int a, int b) {
    return a + b;
} 