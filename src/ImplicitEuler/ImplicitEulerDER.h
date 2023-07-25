#pragma once

#include "ImplicitEuler.h"
#include <src/DER/DER.h>
#include <src/LinearSolver/BaseSolver.h>

#include <Eigen/Dense>

namespace xuan {

class ImplicitEulerDER : public ImplicitEuler {
public:
    ImplicitEulerDER(DER &der, BaseSolver &solver, double h = 0.01) : der_(der), ImplicitEuler(solver, h) {}
    ~ImplicitEulerDER() = default;

    bool projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter = 1000);

    bool advanceOneTimeStep(const std::vector<Eigen::Vector3d> vertices, const std::vector<double> gammas);

private:
    DER &der_;
};

} // namespace xuan