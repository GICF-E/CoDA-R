cmake_minimum_required(VERSION 3.29)
cmake_policy(SET CMP0167 NEW)
set(Boost_DEBUG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Boost CONFIG QUIET COMPONENTS headers)
if(NOT Boost_FOUND)
    # coda-r: header-only Boost (e.g. conda-forge libboost-headers) ships no CMake config; only headers are used
    find_path(BOOST_HEADER_DIR boost/unordered/unordered_flat_map.hpp REQUIRED)
    add_library(Boost::boost INTERFACE IMPORTED)
    target_include_directories(Boost::boost INTERFACE ${BOOST_HEADER_DIR})
    set(Boost_FOUND TRUE)
    set(Boost_INCLUDE_DIRS ${BOOST_HEADER_DIR})
endif()
message(STATUS "Found boost: ${Boost_FOUND} ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

if(MSVC)
    # coda-r: MSVC build with MSVC's own OpenMP runtime (vcomp); /openmp:llvm would load a second LLVM-family runtime
    # next to torch's libiomp5md and abort. No fast-math, so results stay IEEE-exact
    set(CMAKE_CXX_FLAGS "/openmp /EHsc /permissive- /Zc:__cplusplus /bigobj /utf-8 /W3 /arch:AVX2")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DDEBUG /Od /Zi")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNDEBUG /O2")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wextra -Wfatal-errors -march=native -fPIC")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG -DDEBUG -O2 -fno-omit-frame-pointer -g -fvisibility=hidden")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Ofast -g -fvisibility=hidden")
    endif()
endif()


add_subdirectory(csrc)