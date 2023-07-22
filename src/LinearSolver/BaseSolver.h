#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace xuan {

class BaseSolver {
public:
    BaseSolver() = default;
    virtual ~BaseSolver() = default;

    virtual bool solve(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, int max_iter = 1000, double tol = 1e-6) = 0;
};

} // namespace xuan