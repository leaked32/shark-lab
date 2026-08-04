project(shark CXX)

find_package(Boost REQUIRED COMPONENTS json)
find_package(OpenGL REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_library(OPUSENC_LIBRARY NAMES opusenc REQUIRED)
find_path(OPUS_INCLUDE_DIR NAMES opus.h PATH_SUFFIXES opus REQUIRED)
find_library(OPUSFILE_LIBRARY NAMES opusfile REQUIRED)

add_library(shark STATIC
	src/shark.cpp
	src/media.cpp
)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${OPUS_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	Boost::headers Boost::json
	JPEG::JPEG PNG::PNG
	glfw OpenGL::GL
		${OPUSFILE_LIBRARY}
		${OPUSENC_LIBRARY}
)

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
