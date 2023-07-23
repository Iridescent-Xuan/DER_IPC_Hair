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

#define M_PI 3.14159265358979323846

int main() {

    spdlog::set_level(spdlog::level::info);

    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> gammas;
    std::vector<size_t> DBC_vertices, DBC_gammas;

    int num_vertices = 15;
    for (int i = 0; i < num_vertices; ++i) {
        double x = double(i) / num_vertices;
        double z = -0.5 * sin(M_PI * x);
        vertices.push_back(Eigen::Vector3d(x, 0, z));
    }
    for (int i = 0; i < num_vertices - 1; ++i) {
        gammas.push_back(0.0);
    }
    DBC_vertices.push_back(0);
    DBC_vertices.push_back(num_vertices - 1);
    DBC_gammas.push_back(0);
    DBC_gammas.push_back(num_vertices - 2);

    DER der(vertices, gammas, DBC_vertices, DBC_gammas);
    spdlog::info("DER created! {} vertices, {} DBC vertices, {} DBC gammas", der.numVertices(), der.numDBCVertices(), der.numDBCGammas());

    EigenCongugateGradientSolver solver;
    ImplicitEuler implicit_euler(der, solver);

    std::string output_dir = ROOT_PATH + std::string("/test/DER/BendTwist/output/");
    spdlog::info("Output directory: {}", output_dir);
    der.writeOBJ(output_dir + "bendtwist0.obj");

    spdlog::stopwatch watch;
    int frames = 250;
    double d_gamma = M_PI / 2.0;
    for (int i = 1; i <= frames; ++i) {
        der.getGammas(gammas);
        gammas[0] += d_gamma;
        gammas[num_vertices - 2] -= d_gamma;
        if (!implicit_euler.advanceOneTimeStep(vertices, gammas)) {
            return -1;
        };
        if (i % 10 == 0) {
            spdlog::info("********** Frame {} **********", i);
            der.writeOBJ(output_dir + "bendtwist" + std::to_string(i) + std::string(".obj"));
            double energy_pre, energy_after;
            implicit_euler.getEnergy(energy_pre, energy_after);
            spdlog::info("Energy before: {}, Energy after: {}", energy_pre, energy_after);
            spdlog::info("Time elapsed: {}s", watch);
        }
    }

    spdlog::info("Done!");

    return 0;
}