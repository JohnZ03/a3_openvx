cmake_minimum_required(VERSION 3.21)
project(a3_openvx)

set(CMAKE_CXX_STANDARD 23)

set(OpenCV_DIR "/home/haotian/Downloads/build")
find_package(OpenCV REQUIRED)

add_executable(a3_openvx main.cpp)

set(OPENVX_DIR "/home/haotian/Downloads/OpenVX-sample-impl/install/Linux/x64/Debug")

set(OPENVX_INCLUDES "/home/haotian/Downloads/OpenVX-sample-impl/install/Linux/x64/Debug/include")
set(OPENVX_LIBRARIES "/home/haotian/Downloads/OpenVX-sample-impl/install/Linux/x64/Debug/bin/libopenvx.so")

if(DEFINED OPENVX_INCLUDES)
    message(STATUS "OPENVX_INCLUDES: \t${OPENVX_INCLUDES}")
    include_directories(${OPENVX_INCLUDES})
else()
    message(STATUS "OPENVX_INCLUDES: \t<empty>")
endif()

if(DEFINED OPENVX_LIBRARIES)
    message(STATUS "OPENVX_LIBRARIES: \t${OPENVX_LIBRARIES}")
else()
    message(STATUS "OPENVX_LIBRARIES: \t<empty>")
endif()

target_link_libraries(${PROJECT_NAME} ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)