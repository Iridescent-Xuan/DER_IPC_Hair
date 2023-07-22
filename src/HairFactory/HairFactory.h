#pragma once

#include <src/DER/DER.h>

#include <vector>

#include <Eigen/Dense>

namespace xuan {

class HairFactory {
public:
    HairFactory() = default;

    void generateOneHairSrand(const Eigen::Vector3d &root, const Eigen::Vector3d &center_line, double length, double radius, int num_vertices);

private:
    std::vector<DER> hairs;
};

} // namespace xuan