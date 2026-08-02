project(shark CXX)

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)

add_library(shark STATIC
	src/shark.cpp
	src/media.cpp
)

target_include_directories(shark
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(shark PRIVATE
	JPEG::JPEG PNG::PNG
)

target_compile_options(shark PRIVATE -O2)
