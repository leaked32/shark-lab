project(shark CXX)

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_library(OPUSENC_LIBRARY NAMES opusenc REQUIRED)
find_path(OPUS_INCLUDE_DIR NAMES opus.h PATH_SUFFIXES opus REQUIRED)
find_library(OPUSFILE_LIBRARY NAMES opusfile REQUIRED)

add_library(shark STATIC
	src/shark.cpp
	src/media.cpp
)

target_include_directories(shark
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${OPUS_INCLUDE_DIR}
)

target_link_libraries(shark PRIVATE
	JPEG::JPEG PNG::PNG
		${OPUSFILE_LIBRARY}
		${OPUSENC_LIBRARY}
)

target_compile_options(shark PRIVATE -O2)
