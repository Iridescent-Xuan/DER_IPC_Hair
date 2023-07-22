#include "EigenSolver.h"

#include <spdlog/spdlog.h>

namespace xuan {

bool EigenCongugateGradientSolver::solve(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, int max_iter, double tol) {
    solver_.setMaxIterations(max_iter);
    solver_.setTolerance(tol);

    solver_.compute(A);
    x = solver_.solve(b);
    if (solver_.info() != Eigen::Success) {
        spdlog::error("EigenCongugateGradientSolver failed!");
        return false;
    } else {
        spdlog::debug("EigenCongugateGradientSolver succeeded! iterations: {}, estimated error: {}", solver_.iterations(), solver_.error());
    }

    return true;
}

} // namespace xuan