unset(CMAKE_OSX_SYSROOT CACHE)

cmake_minimum_required(VERSION 3.10)

# Project name and version
project(KMeans VERSION 1.0 LANGUAGES C CXX)

# Set directories
set(SRC_DIR src)
set(INCLUDE_DIR include)
set(BUILD_DIR build)
set(BIN_DIR bin)

# Compiler and flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

# Target ARM architecture explicitly
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64") # Ensures compatibility with ARM on macOS
endif()

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

# Source files
file(GLOB_RECURSE C_SRCS "${SRC_DIR}/*.c")
file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.h")

# Include directories
include_directories(${INCLUDE_DIR})

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${BIN_DIR})

# Add executables
add_executable(kmeans ${C_SRCS})
target_link_libraries(kmeans m)

# Delete main.c from C_SRCS
list(FILTER C_SRCS EXCLUDE REGEX ".*/main\\.c$")

# Custom target for debug mode
add_custom_target(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Building in Debug mode"
)

# Help message
message(STATUS "CMake setup for K-means:")
message(STATUS " - Build main program (Release): cmake --build . --target kmeans")
message(STATUS " - Build main program (Debug): cmake -DCMAKE_BUILD_TYPE=Debug .. && make")
message(STATUS " - Clean: cmake --build . --target clean")
