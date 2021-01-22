cmake_minimum_required(VERSION 3.10)
project(lgbm)

set(CMAKE_CXX_STANDARD 11)

if (WIN32 OR MINGW)
    set (platform_macro "-DWINDOWS")
    set (plugin_file "${CMAKE_CURRENT_LIST_DIR}/PluginLGBM_win.txt")
elseif (UNIX)
    set (platform_macro "-DLINUX")
    set (plugin_file "${CMAKE_CURRENT_LIST_DIR}/PluginLGBM.txt")
    add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
endif ()
message("platform: ${platform_macro}")
add_compile_options("-std=c++11" "-fPIC" ${platform_macro} "-Wall" "-o2" "-g")
add_subdirectory(LightGBM)
include_directories(${CMAKE_CURRENT_LIST_DIR}/LightGBM/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/LightGBM/src)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)

link_directories(${CMAKE_CURRENT_LIST_DIR}/LightGBM)
link_directories(${LIBDOLPHINDB})

add_library(PluginLGBM SHARED ./src/lgbm.cpp)

target_link_libraries(PluginLGBM
    DolphinDB 
    _lightgbm
    )

add_custom_command( TARGET PluginLGBM POST_BUILD 
                    COMMAND ${CMAKE_COMMAND} -E copy ${plugin_file} ${CMAKE_BINARY_DIR}/PluginLGBM.txt
                    COMMENT "copy PluginLGBM.txt")


