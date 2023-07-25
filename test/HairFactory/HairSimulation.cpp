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

int main() {

    spdlog::set_level(spdlog::level::debug);

    // generate straight hair from a unit sphere
    HairFactory hair_factory;
    hair_factory.generateHair();
    spdlog::info("{} hairs generated!", hair_factory.numHairs());

    EigenCongugateGradientSolver solver;
    ImplicitEulerHairFactory implicit_euler(hair_factory, solver);

    std::string output_dir = ROOT_PATH + std::string("/test/HairFactory/output/");
    spdlog::info("Output directory: {}", output_dir);
    hair_factory.writeOBJ(output_dir + "hairsim0.obj");

    Eigen::MatrixXd head;
    hair_factory.getHeadVertices(head);

    spdlog::stopwatch watch;
    int frames = 100;
    for (int i = 1; i <= frames; ++i) {
        if (!implicit_euler.advanceOneTimeStep(head)) {
            return -1;
        };
        if (i % 2 == 0) {
            spdlog::info("********** Frame {} **********", i);
            hair_factory.writeOBJ(output_dir + "hairsim" + std::to_string(i) + std::string(".obj"));
            double energy_pre, energy_after;
            implicit_euler.getEnergy(energy_pre, energy_after);
            spdlog::info("Energy before: {}, Energy after: {}", energy_pre, energy_after);
            spdlog::info("Time elapsed: {}s", watch);
        }
    }

    spdlog::info("Done!");

    return 0;
}