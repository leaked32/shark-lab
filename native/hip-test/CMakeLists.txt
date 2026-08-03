project(hip-test CXX HIP)

find_package(Boost REQUIRED COMPONENTS json)

add_executable(hip-test
	main.cpp mhip.hip
)

target_link_libraries(hip-test 
	PRIVATE shark Boost::json
)

target_compile_options(hip-test PRIVATE -O2)
