#include "ImplicitEuler.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/fmt/ostr.h>

#include <fstream>

namespace xuan {

bool ImplicitEuler::projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter) {
    spdlog::debug("Enter projected Newton loop...");
    assert(xt.size() == der_.numVariables());

    Eigen::VectorXd x = xt;
    Eigen::VectorXd x_pre = x;
    double energy;
    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> hess;
    der_.incrementalPotential(x, h_, energy, grad, hess);
    double energy_pre = energy;
    energy_pre_ = energy_pre;
    int iter;
    spdlog::stopwatch watch;
    for (iter = 0; iter < max_iter; ++iter) {
        spdlog::debug("Iteration {}, Energy: {}", iter, energy);

        Eigen::VectorXd p;
        spdlog::debug("----Find search direction...");

        if (!solver_.solve(hess, -grad, p)) {
            spdlog::error("----Failed to solve linear system!");
            return false;
        }

        spdlog::debug("----Line search...");
        double alpha = 1.0;
        while (alpha > 1e-6) {
            x = x_pre + alpha * p;
            der_.incrementalPotential(x, h_, energy, grad, hess);
            if (energy < energy_pre) {
                break;
            }
            alpha *= 0.5;
        }

        energy_pre = energy;
        x_pre = x;
        if (grad.lpNorm<Eigen::Infinity>() < tol) {
            spdlog::debug("----Converged!");
            break;
        }
    }
    if (iter == max_iter) {
        spdlog::warn("----Max iteration reached!");
    }

    spdlog::debug("Exit projected Newton loop, time elapsed: {}s", watch);

    xtt = x;
    energy_after_ = energy;

    return true;
}

bool ImplicitEuler::advanceOneTimeStep(const std::vector<Eigen::Vector3d> vertices, const std::vector<double> gammas) {
    assert(vertices.size() == der_.numVertices() && gammas.size() == der_.numEdges());
    Eigen::VectorXd xt = Eigen::VectorXd::Zero(der_.numVariables());
    for (int i = 0; i < vertices.size(); ++i) {
        xt.segment<3>(3 * i) = vertices[i];
    }
    for (int i = 0; i < gammas.size(); ++i) {
        xt(3 * der_.numVertices() + i) = gammas[i];
    }

    Eigen::VectorXd xtt;
    Eigen::Vector3d bbox = der_.bboxSize();
    double tol = 1e-6 * bbox.squaredNorm() * h_ * h_;
    if (!projectedNewton(xt, xtt, std::sqrt(tol))) {
        spdlog::error("Failed to advance for the next time step!");
        return false;
    }

    der_.deepUpdate(xtt, h_);

    return true;
}

} // namespace xuan