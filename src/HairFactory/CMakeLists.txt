set(target_name "HairFactory")
add_library(${target_name} STATIC)
add_library("xuan::${target_name}" ALIAS ${target_name})

target_compile_features(${target_name}
    PUBLIC
    cxx_std_17
)
target_sources(${target_name}
    PUBLIC
    HairFactory.cpp
)
target_include_directories(${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(${target_name}
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    
    xuan::DER
)