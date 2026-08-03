cmake_minimum_required(VERSION 3.31)
project(chatty CXX)

find_package(OpenGL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS json)
find_package(glfw3 REQUIRED)

project(shark-declick CXX)
add_executable(shark-declick
	declick.cpp
)

# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} .)
target_link_libraries(shark-declick PRIVATE shark)

