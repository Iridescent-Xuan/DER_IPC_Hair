#include <DER/DER.h>
#include <path.h>
#include <ImplicitEuler/ImplicitEuler.h>
#include <LinearSolver/EigenSolver.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

using namespace Eigen;
using namespace std;
using namespace xuan;

int main() {

    spdlog::set_level(spdlog::level::info);

    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> gammas;
    std::vector<size_t> DBC_vertices;

    int num_vertices = 15;
    for (int i = 0; i < num_vertices; ++i) {
        vertices.push_back(Eigen::Vector3d(double(i) / num_vertices, 0, 0));
    }
    for (int i = 0; i < num_vertices - 1; ++i) {
        gammas.push_back(0.0);
    }
    DBC_vertices.push_back(0);

    DER der(vertices, gammas, DBC_vertices, std::vector<size_t>());
    spdlog::info("DER created! {} vertices, {} DBC vertices, {} DBC gammas", der.numVertices(), der.numDBCVertices(), der.numDBCGammas());

    EigenCongugateGradientSolver solver;
    ImplicitEuler implicit_euler(der, solver);

    std::string output_dir = ROOT_PATH + std::string("/test/DER/Gravity/output/");
    spdlog::info("Output directory: {}", output_dir);
    der.writeOBJ(output_dir + "gravity0.obj");

    spdlog::stopwatch watch;
    int frames = 200;
    for (int i = 1; i <= frames; ++i) {
        if (!implicit_euler.advanceOneTimeStep(vertices, gammas)) {
            return -1;
        };
        if (i % 10 == 0) {
            spdlog::info("********** Frame {} **********", i);
            der.writeOBJ(output_dir + "gravity" + std::to_string(i) + std::string(".obj"));
            double energy_pre, energy_after;
            implicit_euler.getEnergy(energy_pre, energy_after);
            spdlog::info("Energy before: {}, Energy after: {}", energy_pre, energy_after);
            spdlog::info("Time elapsed: {}s", watch);
        }
    }

    spdlog::info("Done!");

    return 0;
}