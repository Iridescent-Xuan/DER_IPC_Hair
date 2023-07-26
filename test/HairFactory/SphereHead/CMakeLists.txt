set(target_name "SphereHead")
add_executable(${target_name})
target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_sources(${target_name}
    PRIVATE
    SphereHead.cpp
)
target_include_directories(${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog

    xuan::HairFactory 
    xuan::LinearSolver
    xuan::ImplicitEuler
)