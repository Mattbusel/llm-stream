cmake_minimum_required(VERSION 3.16)
project(llm_stream_examples CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CURL REQUIRED)

add_executable(basic_stream examples/basic_stream.cpp)
target_include_directories(basic_stream PRIVATE include)
target_link_libraries(basic_stream CURL::libcurl)
target_compile_options(basic_stream PRIVATE -Wall -Wextra)

add_executable(chat_loop examples/chat_loop.cpp)
target_include_directories(chat_loop PRIVATE include)
target_link_libraries(chat_loop CURL::libcurl)
target_compile_options(chat_loop PRIVATE -Wall -Wextra)
