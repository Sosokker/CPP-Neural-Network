# Set the C and C++ compiler paths
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")

cmake_minimum_required(VERSION 3.10)

project(CPP-Neural-Network)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add the Eigen library directory (assuming Eigen is in the project root)
include_directories(${CMAKE_SOURCE_DIR}/Eigen)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/Activation)
include_directories(${CMAKE_SOURCE_DIR}/src/Loss)
include_directories(${CMAKE_SOURCE_DIR}/src/Layers)
include_directories(${CMAKE_SOURCE_DIR}/src/utils)

set(SOURCES
    main.cpp
    Activation/ActivationFunc.cpp
    Loss/Loss.cpp
    Layers/DenseLayer.cpp
    utils/utils.cpp
    ../include/Activation/Activation.hpp
    ../include/Layers/DenseLayer.hpp
    ../include/Layers/layers.hpp
    ../include/utils/utils.hpp
    ../include/Eigen/Dense
)

# Create the executable
add_executable(NeuralNetwork ${SOURCES})

# Link the Eigen library
target_include_directories(NeuralNetwork PRIVATE ${CMAKE_SOURCE_DIR}/Eigen)

# Set the output directory for the executable
set_target_properties(NeuralNetwork PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
