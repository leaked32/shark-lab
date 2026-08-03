cmake_minimum_required(VERSION 3.25)

project(shark-lab-cpp
	LANGUAGES CXX HIP
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# -------------------------
# ROCm configuration
# -------------------------

if(NOT DEFINED ROCM_PATH)
	set(ROCM_PATH "/opt/rocm")
endif()

list(APPEND CMAKE_PREFIX_PATH
	${ROCM_PATH}
)


# -------------------------
# shark library
# -------------------------

add_subdirectory("shark")

# -------------------------
# HIP test
# -------------------------

add_subdirectory("hip-test")

# Applications

add_subdirectory("chatty")
