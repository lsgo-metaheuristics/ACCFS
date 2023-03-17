cmake_minimum_required(VERSION 3.24)
project(PAR_FS_VSCODE)

set(CMAKE_CXX_STANDARD 17)

add_executable(PAR_FS_VSCODE main.cpp decomposer.cpp matrix.cpp minmaxscaler.cpp SHADE.cpp CCDE.cpp)
