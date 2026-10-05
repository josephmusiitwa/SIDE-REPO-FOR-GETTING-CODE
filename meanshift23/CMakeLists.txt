cmake_minimum_required(VERSION 3.15)

# 
project(MeanShift
    VERSION 1.0.0
    LANGUAGES CXX
)

add_library(meanshift STATIC
    src/Dataset.cpp
    src/MeanShift.cpp
)

target_include_directories(meanshift
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(meanshift PUBLIC cxx_std_23)

install(
    TARGETS meanshift
    EXPORT MeanShiftTargets
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY include/meanshift
    DESTINATION include
)

install(
    EXPORT MeanShiftTargets
    FILE MeanShiftTargets.cmake
    NAMESPACE MeanShift::
    DESTINATION lib/cmake/MeanShift
)

# Add Examples and Tests from subdirectories
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)
