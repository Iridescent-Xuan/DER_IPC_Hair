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

    int num_vertices = 20;
    // cylindrical helix
    double radius = 0.1;
    double round = 6.0 * M_PI;
    for (int i = 0; i < num_vertices; ++i) {
        double theta = double(i) / num_vertices;
        vertices.push_back(Eigen::Vector3d(radius * cos(round * theta), radius * sin(round * theta), -theta));
    }
    for (int i = 0; i < num_vertices - 1; ++i) {
        gammas.push_back(0.0);
    }
    DBC_vertices.push_back(0);

    DER der(vertices, gammas, DBC_vertices, std::vector<size_t>());
    spdlog::info("DER created! {} vertices, {} DBC vertices, {} DBC gammas", der.numVertices(), der.numDBCVertices(), der.numDBCGammas());

    der.setParameters(3e3, 6e-3, 1e-2);

    EigenCongugateGradientSolver solver;
    ImplicitEuler implicit_euler(der, solver);

    std::string output_dir = ROOT_PATH + std::string("/test/DER/Curly/output/");
    spdlog::info("Output directory: {}", output_dir);
    der.writeOBJ(output_dir + "curly0.obj");

    spdlog::stopwatch watch;

    // translate
    Eigen::Vector3d speed;
    int frame = 0;
    for (int i = 1; i <= 3; ++i) {
        der.getVertices(vertices);
        switch (i) {
        case 1:
            speed = Eigen::Vector3d(0.05, 0, 0);
            break;
        case 2:
            speed = Eigen::Vector3d(0, 0.05, 0);
            break;
        case 3:
            speed = Eigen::Vector3d(0, 0, 0.05);
            break;
        }
        for (int j = 0; j < 20; ++j) {
            spdlog::info("********** Frame {} **********", ++frame);
            vertices[0] += speed;
            if (!implicit_euler.advanceOneTimeStep(vertices, gammas)) {
                return -1;
            };
            der.writeOBJ(output_dir + "curly" + std::to_string(frame) + std::string(".obj"));
            double energy_pre, energy_after;
            implicit_euler.getEnergy(energy_pre, energy_after);
            spdlog::info("Energy before: {}, Energy after: {}", energy_pre, energy_after);
            spdlog::info("Time elapsed: {}s", watch);
        }
    }

    spdlog::info("Done!");

    return 0;
}