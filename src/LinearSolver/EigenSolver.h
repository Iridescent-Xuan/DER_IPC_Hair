#pragma once

#include "BaseSolver.h"

#include <Eigen/IterativeLinearSolvers>

namespace xuan {

class EigenCongugateGradientSolver : public BaseSolver{
public:
    EigenCongugateGradientSolver() = default;
    ~EigenCongugateGradientSolver() = default;

    bool solve(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, int max_iter = 1000, double tol = 1e-6);

private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver_;
};

} // namespace xuan