#include <src/LinearSolver/BaseSolver.h>
namespace xuan {

class ImplicitEuler {
public:
    ImplicitEuler(BaseSolver &solver, double h) : solver_(solver), h_(h) {}
    ~ImplicitEuler() = default;

    virtual bool projectedNewton(const Eigen::VectorXd &xt, Eigen::VectorXd &xtt, double tol, int max_iter) = 0;

    void getEnergy(double &energy_pre, double &energy_after) const {
        energy_pre = energy_pre_;
        energy_after = energy_after_;
    }

protected:
    BaseSolver &solver_;
    double h_; // time step size
    double energy_pre_ = 0.0;
    double energy_after_ = 0.0;
};

} // namespace xuan