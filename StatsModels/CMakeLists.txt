cmake_minimum_required(VERSION 3.17)
project(statsmodels)

set(CMAKE_CXX_STANDARD 11)

ADD_DEFINITIONS(-DLINUX)
ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0)
add_compile_options(-fPIC)
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
link_directories(${LIBDOLPHINDB})
link_libraries(DolphinDB)

add_library(PluginStatsmodels SHARED ./src/statsmodels.cpp)

