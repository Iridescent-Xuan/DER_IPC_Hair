#pragma once

#include <src/DER/DER.h>
#include <src/LinearSolver/BaseSolver.h>

#include <Eigen/Dense>

namespace xuan {

class ImplicitEuler {
public:
    ImplicitEuler(DER &der, BaseSolver &solver, double h = 0.01) : der_(der), solver_(solver), h_(h) {}
    ~ImplicitEuler() = default;

    bool projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter = 1000);

    bool advanceOneTimeStep(const std::vector<Eigen::Vector3d> vertices, const std::vector<double> gammas);

    void getEnergy(double &energy_pre, double &energy_after) const {
        energy_pre = energy_pre_;
        energy_after = energy_after_;
    }

private:
    DER &der_;
    BaseSolver &solver_;
    double h_; // time step size
    double energy_pre_ = 0.0;
    double energy_after_ = 0.0;
};

} // namespace xuan