set(target_name "test")
add_executable(${target_name})
target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_sources(${target_name}
    PRIVATE
    test.cpp
)
target_include_directories(${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen

    DER
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/path.h.in ${CMAKE_CURRENT_SOURCE_DIR}/path.h)