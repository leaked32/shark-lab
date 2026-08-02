project(shark CXX)

add_library(shark STATIC
	src/shark.cpp
)

target_include_directories(shark
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(shark PRIVATE -O2)
