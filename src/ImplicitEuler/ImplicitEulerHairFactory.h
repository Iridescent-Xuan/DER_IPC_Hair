#pragma once

#include "ImplicitEuler.h"
#include <src/HairFactory/HairFactory.h>
#include <src/LinearSolver/BaseSolver.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace xuan {

class ImplicitEulerHairFactory : public ImplicitEuler {
public:
    ImplicitEulerHairFactory(HairFactory &hair_factory, BaseSolver &solver, double h = 0.01) : hair_factory_(hair_factory), ImplicitEuler(solver, h) {}
    ~ImplicitEulerHairFactory() = default;

    bool projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter = 1000);

    bool advanceOneTimeStep(const Eigen::MatrixXd &head);

private:
    HairFactory &hair_factory_;
    Eigen::MatrixXd head_;
};

} // namespace xuan