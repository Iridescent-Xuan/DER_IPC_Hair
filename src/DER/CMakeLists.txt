set(target_name "DER")
add_library(${target_name} STATIC) # SHARED for shared library loading; STATIC for static library loading
target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_sources(${target_name}
    PUBLIC
    DER.cpp
)
target_include_directories(${target_name}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(${target_name}
    PUBLIC
    # -fPIC # for shared library loading
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    LinearSolver
)