cmake_minimum_required(VERSION 3.8)

project(mnist CXX)

add_executable(${PROJECT_NAME}
    mnist.cpp
)

set(MYNN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
# my mac can't find OpenBLAS :(
if (APPLE)
    set(BLAS_INCLUDE_DIRS /usr/local/opt/openblas/include/)
    set(BLAS_LIBS /usr/local/opt/openblas/lib/)
else ()
    find_package(BLAS REQUIRED)
endif ()
find_package(OpenMP REQUIRED)

target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_14
)
target_compile_options(${PROJECT_NAME}
    PUBLIC -Wall -Wextra
)
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${MYNN_INCLUDE_DIRS}
    ${BLAS_INCLUDE_DIRS}
    ${OpenMP_INCLUDE_DIRS}
)
if (APPLE)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${BLAS_LIBS}/libopenblas.dylib
        ${OpenMP_LIBS}
    )
else ()
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${BLAS_LIBS}
        ${OpenMP_LIBS}
    )
endif ()
