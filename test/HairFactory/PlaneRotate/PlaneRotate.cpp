#include <HairFactory/HairFactory.h>
#include <path.h>
#include <ImplicitEuler/ImplicitEulerHairFactory.h>
#include <LinearSolver/EigenSolver.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <Eigen/Dense>

#include <string>

using namespace Eigen;
using namespace std;
using namespace xuan;

Eigen::MatrixXd rotate(const Eigen::MatrixXd &V, const Eigen::Vector3d &axis, double angle) {
    Eigen::MatrixXd V_rotated(V.rows(), V.cols());
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    for (int i = 0; i < V.rows(); ++i) {
        V_rotated.row(i) = R * V.row(i).transpose();
    }
    return V_rotated;
}

int main() {

    spdlog::set_level(spdlog::level::debug);

    // generate curly hair from a plane (-0.1, -0.1, 0) × (0.1, 0.1, 0)
    int x_resolution = 3;
    int y_resolution = 3;
    double x_min = -0.1;
    double x_max = 0.1;
    double y_min = -0.1;
    double y_max = 0.1;
    double z = 0.0;
    Eigen::MatrixXd V(x_resolution * y_resolution, 3);
    Eigen::MatrixXi F((x_resolution - 1) * (y_resolution - 1) * 2, 3);
    Eigen::MatrixXd N(x_resolution * y_resolution, 3);

    for (int i = 0; i < x_resolution; ++i) {
        for (int j = 0; j < y_resolution; ++j) {
            V.row(i * y_resolution + j) = Eigen::Vector3d(x_min + (x_max - x_min) / (x_resolution - 1) * i, y_min + (y_max - y_min) / (y_resolution - 1) * j, z);
            N.row(i * y_resolution + j) = Eigen::Vector3d(0.0, 0.0, -1.0);
        }
    }

    for (int i = 0; i < x_resolution - 1; ++i) {
        for (int j = 0; j < y_resolution - 1; ++j) {
            F.row((i * (y_resolution - 1) + j) * 2) = Eigen::Vector3i(i * y_resolution + j, i * y_resolution + j + 1, (i + 1) * y_resolution + j);
            F.row((i * (y_resolution - 1) + j) * 2 + 1) = Eigen::Vector3i(i * y_resolution + j + 1, (i + 1) * y_resolution + j + 1, (i + 1) * y_resolution + j);
        }
    }

    HairFactory hair_factory;
    double radius = 0.1;
    double curliness = 6.0;
    hair_factory.generateHair(V, F, N, radius, curliness);
    spdlog::info("{} hairs generated!", hair_factory.numHairs());

    EigenCongugateGradientSolver solver;
    ImplicitEulerHairFactory implicit_euler(hair_factory, solver);

    std::string output_dir = ROOT_PATH + std::string("/test/HairFactory/PlaneRotate/output/");
    spdlog::info("Output directory: {}", output_dir);
    hair_factory.writeOBJ(output_dir + "plane0.obj");

    Eigen::MatrixXd head;
    hair_factory.getHeadVertices(head);

    spdlog::stopwatch watch;
    int frames = 100;
    double angular_speed = M_PI / 10;
    for (int i = 1; i <= frames; ++i) {
        head = rotate(head, Eigen::Vector3d(0.0, 0.0, 1.0), angular_speed);
        if (!implicit_euler.advanceOneTimeStep(head)) {
            return -1;
        };
        if (i % 2 == 0) {
            spdlog::info("********** Frame {} **********", i);
            hair_factory.writeOBJ(output_dir + "plane" + std::to_string(i) + std::string(".obj"));
            double energy_pre, energy_after;
            implicit_euler.getEnergy(energy_pre, energy_after);
            spdlog::info("Energy before: {}, Energy after: {}", energy_pre, energy_after);
            spdlog::info("Time elapsed: {}s", watch);
        }
    }

    spdlog::info("Done!");

    return 0;
}