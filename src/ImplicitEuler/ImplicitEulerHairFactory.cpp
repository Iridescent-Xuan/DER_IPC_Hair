#include "ImplicitEulerHairFactory.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <vector>

namespace xuan {

bool ImplicitEulerHairFactory::projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter) {
    spdlog::debug("Enter projected Newton loop...");
    assert(xt.size() == hair_factory_.numVariables());

    Eigen::VectorXd x = xt;
    Eigen::VectorXd x_pre = x;
    double energy;
    std::vector<Eigen::VectorXd> gradients;
    std::vector<Eigen::SparseMatrix<double>> hessians;
    hair_factory_.accumulateDerivatives(x, head_, h_, energy, gradients, hessians);
    double energy_pre = energy;
    energy_pre_ = energy_pre;
    int iter;
    spdlog::stopwatch watch;
    for (iter = 0; iter < max_iter; ++iter) {
        spdlog::debug("Iteration {}, Energy: {}", iter, energy);

        Eigen::VectorXd p;
        p.resize(hair_factory_.numVariables());
        spdlog::debug("----Find search direction...");
        std::vector<size_t> offset = hair_factory_.hairOffset();
        for (size_t h = 0; h < hair_factory_.numHairs(); ++h) {
            Eigen::VectorXd p_i;
            if (!solver_.solve(hessians[h], -gradients[h], p_i)) {
                spdlog::error("----Failed to solve linear system for hair {}!", h);
                return false;
            }
            p.segment(offset[h], hair_factory_.numVariablesPerHair(h)) = p_i;
        }

        spdlog::debug("----Line search...");
        double alpha = hair_factory_.computeCollisionFreeStepSize(x, x + p, head_);
        spdlog::debug("----Largest step size: {}", alpha);
        while (alpha > 1e-6) {
            x = x_pre + alpha * p;
            hair_factory_.accumulateDerivatives(x, head_, h_, energy, gradients, hessians);
            if (energy < energy_pre) {
                break;
            }
            alpha *= 0.5;
        }
        spdlog::debug("----Alpha: {}", alpha);

        if(fabs(energy - energy_pre) / energy < 1e-6) {
            spdlog::debug("----Converged! energy: {}", energy);
            break;
        }

        energy_pre = energy;
        x_pre = x;
        double p_norm = p.lpNorm<Eigen::Infinity>();
        spdlog::debug("----Norm of p: {}", p_norm);
        if (p_norm < tol) {
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

    if (hair_factory_.violateDBC(xt, xtt)) {
        spdlog::error("DBC violated!");
        return false;
    }

    return true;
}

bool ImplicitEulerHairFactory::advanceOneTimeStep(const Eigen::MatrixXd &head) {
    head_ = head;
    Eigen::VectorXd xt = hair_factory_.moveHead(head);

    Eigen::Vector3d bbox = hair_factory_.hairBoundingBox();
    double tol = 1e-4 * bbox.squaredNorm();

    Eigen::VectorXd xtt;
    if (!projectedNewton(xt, xtt, std::sqrt(tol))) {
        spdlog::error("Failed to advance for the next time step!");
        return false;
    }

    hair_factory_.deepUpdate(xtt, h_, head);
    return true;
}

} // namespace xuan