cmake_minimum_required(VERSION 3.22)
project(render LANGUAGES CXX)

option(RENDER_ENABLE_CUDA "Build CUDA backend when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(render_cpu
    Backend/CPU/main.cpp
)
target_include_directories(render_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sample_generator_test
    Tests/SampleGeneratorTest.cpp
)
target_include_directories(sample_generator_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME sample_generator COMMAND sample_generator_test)

add_executable(surface_identity_test
    Tests/SurfaceIdentityTest.cpp
)
target_include_directories(surface_identity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME surface_identity COMMAND surface_identity_test)

add_executable(reconstruction_test
    Tests/ReconstructionTest.cpp
)
target_include_directories(reconstruction_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME reconstruction COMMAND reconstruction_test)

if(MSVC)
    target_compile_options(render_cpu PRIVATE /O2)
else()
    target_compile_options(render_cpu PRIVATE -O3)
    find_package(Threads REQUIRED)
    target_link_libraries(render_cpu PRIVATE Threads::Threads)
endif()

if(RENDER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        add_executable(render_cuda
            Backend/CUDA/main.cu
        )
        target_include_directories(render_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(render_cuda PRIVATE CUDA::cudart)
        set_target_properties(render_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "89"
        )
    else()
        message(STATUS "CUDA compiler not found; render_cuda target will not be built")
    endif()
endif()
