cmake_minimum_required(VERSION 3.14)
project(llm-stream CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
set(CURL_LIB ${CURL_LIBRARIES})
set(CURL_INC ${CURL_INCLUDE_DIRS})

foreach(EXAMPLE basic_stream chat_loop responses_stream anthropic_stream)
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)
    target_include_directories(${EXAMPLE} PRIVATE include ${CURL_INC})
    target_link_libraries(${EXAMPLE} PRIVATE ${CURL_LIB})
    target_compile_options(${EXAMPLE} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    )
endforeach()
# Offline test of the SSE parsers (no network, no API key).
enable_testing()
add_executable(test_responses_sse tests/test_responses_sse.cpp)
target_include_directories(test_responses_sse PRIVATE include ${CURL_INC})
target_link_libraries(test_responses_sse PRIVATE ${CURL_LIB})
add_test(NAME test_responses_sse COMMAND test_responses_sse)

# SSE parser on its own (LLM_STREAM_NO_CURL): no libcurl on the include path or link line.
add_executable(test_sse tests/test_sse.cpp)
target_include_directories(test_sse PRIVATE include)
add_test(NAME test_sse COMMAND test_sse)
