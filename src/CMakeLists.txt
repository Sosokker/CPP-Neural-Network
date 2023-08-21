cmake_minimum_required(VERSION 3.10)

project(cpp_neural_network)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source files
set(SOURCES
    src/Model/Model.cpp
    src/utils/Accuracy.cpp
    src/Optimizer/Optimizer.cpp
    src/Loss/Loss.cpp
    src/Layers/Layers.cpp
    src/ActivationFunction/ActivationFunction.cpp
    main.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Layers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Activation
)
