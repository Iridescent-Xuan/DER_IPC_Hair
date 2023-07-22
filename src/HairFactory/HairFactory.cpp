#include "HairFactory.h"

namespace xuan {
void HairFactory::generateOneHairSrand(const Eigen::Vector3d &root, const Eigen::Vector3d &center_line, double length, double radius, int num_vertices) {
    assert(num_vertices >= 2 && radius >= 0 && length > 0);

    std::vector<Eigen::Vector3d> vertices(num_vertices);
    if (radius < 1e-6) { // straight hair
        double step = length / (num_vertices - 1);
        for (int i = 0; i < num_vertices; ++i) {
            vertices[i] = root + i * step * center_line.normalized();
        }
    } else { // curly hair
    }

    std::vector<double> gammas(num_vertices - 1, 0.0);
    std::vector<size_t> DBC_vertices{0};
    
    hairs.emplace_back(DER(vertices, gammas, DBC_vertices, std::vector<size_t>()));
}

} // namespace xuan