cmake_minimum_required(VERSION 3.21)

project(hip_test LANGUAGES CXX HIP)

add_executable(hip_test main.hip)

set_target_properties(hip_test PROPERTIES
    HIP_ARCHITECTURES gfx1103
)
