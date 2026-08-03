cmake_minimum_required(VERSION 3.31)
project(chatty CXX)

find_package(OpenGL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS json)
find_package(glfw3 REQUIRED)

include_directories(
	imgui
	imgui/backends
	${GLFW_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}
	db.cpp
	discrete.cpp
	launcher.cpp
	external.cpp
	
	imgui/imgui.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_widgets.cpp
	imgui/imgui_tables.cpp
	imgui/backends/imgui_impl_glfw.cpp
	imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} .)
target_link_libraries(${PROJECT_NAME} PRIVATE
	shark
	SQLite3::SQLite3
	Boost::headers Boost::json
	glfw OpenGL::GL
)

