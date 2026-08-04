project(hip-test CXX HIP)

find_package(Boost REQUIRED COMPONENTS json)

add_executable(${PROJECT_NAME}
	main.cpp mhip.hip
)

target_link_libraries(${PROJECT_NAME}
	PRIVATE shark Boost::json
)

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
