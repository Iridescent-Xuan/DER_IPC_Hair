#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <spdlog/spdlog.h>

namespace xuan {

// direct solver
class EigenSimplicialLDLT {
public:
    EigenSimplicialLDLT() : solver_() {
    }

    bool init(const Eigen::SparseMatrix<double> &A) {
        solver_.compute(A);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenSimplicialLDLT decomposition failed!");
            return false;
        }
        return true;
    }

    bool solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) {
        x = solver_.solve(b);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenSimplicialLDLT solve failed!");
            x = Eigen::VectorXd::Zero(b.size());
            return false;
        }
        return true;
    }

private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;
};

// iterative solver
class EigenBiCGSTAB {
public:
    EigenBiCGSTAB(int max_iter = 1000, double tolerance = 1e-6) : solver_() {
        solver_.setMaxIterations(max_iter);
        solver_.setTolerance(tolerance);
    }

    bool init(const Eigen::SparseMatrix<double> &A) {
        solver_.compute(A);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenBiCGSTAB decomposition failed!");
            return false;
        }
        return true;
    }

    bool solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) {
        x = solver_.solveWithGuess(b, x);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenBiCGSTAB solve failed!");
            x = Eigen::VectorXd::Zero(b.size());
            spdlog::error("EigenBiCGSTAB: error: {}", solver_.error());
            return false;
        }
        spdlog::info("EigenBiCGSTAB: #iterations: {}, estimated error: {}", solver_.iterations(), solver_.error());
        return true;
    }

private:
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_;
};

class EigenConjugateGradient {
public:
    EigenConjugateGradient(int max_iter = 1000, double tolerance = 1e-6) : solver_() {
        solver_.setMaxIterations(max_iter);
        solver_.setTolerance(tolerance);
    }

    bool init(const Eigen::SparseMatrix<double> &A) {
        solver_.compute(A);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenConjugateGradient decomposition failed!");
            return false;
        }
        return true;
    }

    bool solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) {
        x = solver_.solve(b);
        if (solver_.info() != Eigen::Success) {
            spdlog::error("EigenConjugateGradient solve failed!");
            x = Eigen::VectorXd::Zero(b.size());
            spdlog::error("EigenConjugateGradient: error: {}", solver_.error());
            return false;
        }
        spdlog::info("EigenConjugateGradient: #iterations: {}, estimated error: {}", solver_.iterations(), solver_.error());
        return true;
    }

private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver_;
};

} // namespace xuan