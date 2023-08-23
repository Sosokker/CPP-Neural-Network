cmake_minimum_required(VERSION 3.10)

project(CPP-Neural-Network)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Set the directories for header files and source files
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/include/Activation)
include_directories(${CMAKE_SOURCE_DIR}/include/Layers)

# Add source files to the project
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create the executable
add_executable(CPP-Neural-Network ${SOURCES})

# Link the Eigen library
target_include_directories(CPP-Neural-Network PRIVATE ${CMAKE_SOURCE_DIR}/Eigen)

# Set the output directory for the executable
set_target_properties(CPP-Neural-Network PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
