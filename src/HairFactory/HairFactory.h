#pragma once

#include <src/DER/DER.h>

#include <vector>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <ipc/ipc.hpp>

namespace xuan {

class HairFactory {
public:
    HairFactory() = default;

    // TODO: generate hair from a given obj file
    void generateHair(const std::string &obj_path) {}

    // generate hair from a unit sphere
    void generateHair(double radius = 0.0, double curliness = 0.0, int vertices_per_strand = 15, double length = 1.0, int resolution_theta = 20, int resolution_phi = 10);

    // output
    void writeOBJ(const std::string &filename) const;

    void getHeadVertices(Eigen::MatrixXd &V) const {
        V = V_head_;
    }

    size_t numVariables() const {
        return num_variables_;
    }

    size_t numHairs() const {
        return hairs_.size();
    }

    size_t numVariablesPerHair(size_t i) const {
        return hairs_[i].numVariables();
    }

    std::vector<size_t> hairOffset() const {
        std::vector<size_t> offset(hairs_.size());
        offset[0] = 0;
        for (size_t i = 1; i < hairs_.size(); ++i) {
            offset[i] = offset[i - 1] + hairs_[i - 1].numVariables();
        }
        return offset;
    }

    Eigen::Vector3d hairBoundingBox() const;

    // update one hair strand
    void deepUpdateOneHairStrand(const Eigen::VectorXd &x, size_t index_hair, double h);

    // collision barrier
    // we should pass the head since it is not updated yet
    double collisionEnergy(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints = true);
    // return: gradient (vertices + gammas) for each hair strand
    std::vector<Eigen::VectorXd> collisionGradient(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints = true);
    // return: hessian (vertices + gammas)^2 for each hair strand
    std::vector<Eigen::SparseMatrix<double>> collisionHessian(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints = true);

    // largest step size that is collision free
    double computeCollisionFreeStepSize(const Eigen::VectorXd &xt, const Eigen::VectorXd &xtt, const Eigen::MatrixXd &head);

    // return the DER variables for this time step (xt)
    Eigen::VectorXd moveHead(const Eigen::MatrixXd &head);

    // energy and derivatives for the solver
    void accumulateDerivatives(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, double h,
                               /* output */
                               double &energy, std::vector<Eigen::VectorXd> &grad, std::vector<Eigen::SparseMatrix<double>> &hess);

    void deepUpdate(const Eigen::VectorXd &x, double h, const Eigen::MatrixXd &head);

    bool violateDBC(const Eigen::VectorXd &xt, const Eigen::VectorXd &xtt);

private:
    // helper functions

    void generateOneHairSrand(const Eigen::Vector3d &root, const Eigen::Vector3d &center_line, int num_vertices = 15, double length = 1.0, double radius = 0.0, double curliness = 0.0);

    void generateCollisionMesh();

    // x: all hair variables (vertices + gammas)
    // return: collision vertices (head + hair)
    Eigen::MatrixXd generateCollisionVertices(const Eigen::VectorXd &x, const Eigen::MatrixXd &head);

private:
    std::vector<DER> hairs_;
    std::vector<size_t> hair_roots_; // index of the root vertex of each hair
    double dhat_;                    // collision barrier

    // head mesh
    // triangles only
    Eigen::MatrixXd V_head_;
    Eigen::MatrixXi F_head_;
    Eigen::MatrixXi E_head_;

    // collision attributes
    ipc::CollisionMesh collision_mesh_;
    ipc::CollisionConstraints collision_constraints_;

    // constants
    double kappa_ = 1e-4;      // collision stiffness
    size_t num_variables_ = 0; // all hair variables (vertices + gammas)
    size_t num_hair_vertices_ = 0;
    size_t num_hair_edges_ = 0;
};

} // namespace xuan