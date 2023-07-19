#include "DER/DER.h"
#include "path.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>

#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace xuan;

int main() {
    std::vector<Vector3d> vertices;
    for (int i = 0; i < 10; ++i) {
        vertices.emplace_back(i / 10.0, 0, 0);
    }

    std::vector<int> DBC_v_index = {0, vertices.size() - 1};
    std::vector<Vector3d> DBC_v = {vertices[0], vertices[vertices.size() - 1]};
    std::vector<int> DBC_gamma_index = {0, vertices.size() - 2};
    std::vector<double> DBC_gamma = {0, 0};

    std::vector<double> mass(vertices.size(), 1.0);

    DER der(vertices, DBC_v_index, DBC_gamma_index, mass);

    for (int i = 0; i < 100; ++i) {
        spdlog::info("i = {}", i);
        DBC_gamma[0] = i / 100.0;
        DBC_gamma[1] = -i / 100.0;
        der.applyDBC(DBC_v, DBC_gamma);
        if (!der.solve()) {
            spdlog::error("DER solve failed!");
            return -1;
        }
        der.writeObj(TEST_FILE_DIRECTORY + std::string("/output/test") + std::to_string(i) + ".obj");
    }

    der.printGamma();
    der.printFrames();

    spdlog::info("DER solve success!");

    return 0;
}