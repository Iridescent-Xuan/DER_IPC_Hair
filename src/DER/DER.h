#pragma once

#include <vector>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace xuan {

class DER {
public:
    DER(const std::vector<Eigen::Vector3d> vertices, const std::vector<double> gammas, const std::vector<size_t> DBC_vertices, const std::vector<size_t> DBC_gammas);

    // stretch
    double stretchEnergy(const std::vector<Eigen::Vector3d> &vertices);
    void stretchGradient(const std::vector<Eigen::Vector3d> &vertices, /* output */ Eigen::VectorXd &gradient);                            // gradient: 3 * #vertices + #gammas
    void stretchHessian(const std::vector<Eigen::Vector3d> &vertices, /* output */ std::vector<Eigen::Triplet<double>> &hessian_triplets); // hessian: (3 * #vertices + #gammas)^2

    // bend
    double bendEnergy(const std::vector<Eigen::Vector2d> &kappas);
    void bendGradient(const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                      const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                      /* output */ Eigen::VectorXd &gradient);
    void bendHessian(const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector2d> &kappas,
                     const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                     /* output */ std::vector<Eigen::Triplet<double>> &hessian_triplets);

    // twist
    double twistEnergy(const std::vector<double> &gammas, const std::vector<double> &reference_twists);
    void twistGradient(const std::vector<double> &gammas, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges, const std::vector<double> &reference_twists,
                       /* output */ Eigen::VectorXd &gradient);
    void twistHessian(const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges,
                      /* output */ std::vector<Eigen::Triplet<double>> &hessian_triplets);

    // kinetic
    double kineticEnergy(const std::vector<Eigen::Vector3d> &vertices, double h);
    void kineticGradient(const std::vector<Eigen::Vector3d> &vertices, double h,
                         /* output */ Eigen::VectorXd &gradient);           // gradient: 3 * #vertices + #gammas
    void kineticHessian(/* output */ Eigen::SparseMatrix<double> &hessian); // hessian: (3 * #vertices + #gammas)^2

    // accumulate
    double elasticEnergy(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector2d> &kappas, const std::vector<double> &gammas, const std::vector<double> &reference_twists);
    void elasticGradient(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> &gammas, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                         const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2, const std::vector<double> &reference_twists,
                         /* output */ Eigen::VectorXd &gradient);
    void elasticHessian(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                        const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                        /* output */ Eigen::SparseMatrix<double> &hessian);
    void elasticAttributes(const Eigen::VectorXd &x,
                           /* output */
                           double &elastic_energy, Eigen::VectorXd &elastic_gradient, Eigen::SparseMatrix<double> &elastic_hessian);
    void kineticAttributes(const Eigen::VectorXd &x, double h,
                           /* output */
                           double &kinetic_energy, Eigen::VectorXd &kinetic_gradient, Eigen::SparseMatrix<double> &kinetic_hessian);
    void incrementalPotential(const Eigen::VectorXd &x, double h,
                              /* output */
                              double &energy, Eigen::VectorXd &gradient, Eigen::SparseMatrix<double> &hessian);

    // update the rod status
    void deepUpdate(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> gammas, double h);
    void deepUpdate(const Eigen::VectorXd &x, double h);

    void update(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> gammas,
                /* output */
                std::vector<Eigen::Vector3d> &reference_frames1, std::vector<Eigen::Vector3d> &reference_frames2,
                std::vector<Eigen::Vector3d> &material_frames1, std::vector<Eigen::Vector3d> &material_frames2,
                std::vector<Eigen::Vector3d> &edges, std::vector<double> &reference_twists,
                std::vector<Eigen::Vector3d> &curvatures, std::vector<Eigen::Vector2d> &kappas,
                bool constructor = false);

    // DBC
    void handleDBC(Eigen::VectorXd &gradient, Eigen::SparseMatrix<double> &hessian);

    // getters
    size_t numVariables() const {
        return 3 * num_vertices_ + num_edges_;
    }
    size_t numVertices() const {
        return num_vertices_;
    }
    size_t numEdges() const {
        return num_edges_;
    }
    size_t numDBCVertices() const {
        return DBC_vertices_.size();
    }
    size_t numDBCGammas() const {
        return DBC_gammas_.size();
    }
    void getVertices(std::vector<Eigen::Vector3d> &vertices) const {
        vertices = vertices_;
    }
    Eigen::Vector3d bboxSize() const;

    // output
    void writeOBJ(const std::string &filename) const;

    // set parameters
    void setParameters(const double &ks, const double &kb, const double &kt, const double &mass = 1.0, const Eigen::Vector3d &gravity = Eigen::Vector3d(0.0, 0.0, -9.8)) {
        ks_ = ks;
        kb_ = kb;
        kt_ = kt;
        mass_ = mass;
        gravity_ = gravity;
    }

private:
    // helper functions

    bool validVertexIndex(const size_t &index) const {
        return index >= 0 && index < num_vertices_;
    }
    bool validEdgeIndex(const size_t &index) const {
        return index >= 0 && index < num_edges_;
    }

    Eigen::Vector3d parallelTransport(const Eigen::Vector3d &target, const Eigen::Vector3d &u, const Eigen::Vector3d &v);

    Eigen::Vector3d rotate(const Eigen::Vector3d &v, const Eigen::Vector3d &axis, double angle);

    // local

    double stretchEnergyLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices);
    Eigen::Vector<double, 6> stretchGradientLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices);
    Eigen::Matrix<double, 6, 6> stretchHessianLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices);

    double bendEnergyLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas);
    void kappaGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                            const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                            Eigen::Vector3d &d_kappa1_e_c, Eigen::Vector3d &d_kappa1_e_p, Eigen::Vector3d &d_kappa2_e_c, Eigen::Vector3d &d_kappa2_e_p,
                            double &d_kappa1_gamma_c, double &d_kappa1_gamma_p, double &d_kappa2_gamma_c, double &d_kappa2_gamma_p);
    Eigen::Vector<double, 11> bendGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                                                const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2);
    void kappaHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                           const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                           Eigen::Vector3d &dd_kappa1_ec_gamma_c, Eigen::Vector3d &dd_kappa1_ec_gamma_p, Eigen::Vector3d &dd_kappa1_ep_gamma_c, Eigen::Vector3d &dd_kappa1_ep_gamma_p,
                           Eigen::Vector3d &dd_kappa2_ec_gamma_c, Eigen::Vector3d &dd_kappa2_ec_gamma_p, Eigen::Vector3d &dd_kappa2_ep_gamma_c, Eigen::Vector3d &dd_kappa2_ep_gamma_p,
                           double &dd_kappa1_gamma_c_gamma_c, double &dd_kappa1_gamma_c_gamma_p, double &dd_kappa1_gamma_p_gamma_p,
                           double &dd_kappa2_gamma_c_gamma_c, double &dd_kappa2_gamma_c_gamma_p, double &dd_kappa2_gamma_p_gamma_p
                           /* Note that we ignore the second derivatives w.r.t. edges, i.e. dd_kappa_e_e */);
    Eigen::Matrix<double, 11, 11> bendHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                                                   const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2);

    double twistEnergyLocal(const size_t &index_vertex, const std::vector<double> &gammas, const std::vector<double> &reference_twists);
    void miGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges,
                         Eigen::Vector3d &d_mi_e_c, Eigen::Vector3d &d_mi_e_p, double &d_mi_gamma_c, double &d_mi_gamma_p);
    Eigen::Vector<double, 11> twistGradientLocal(const size_t &index_vertex, const std::vector<double> gammas, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges, const std::vector<double> &reference_twists);
    Eigen::Matrix<double, 11, 11> twistHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges);

    // map from local to global

    void fillGradient(const Eigen::VectorXd &gradient_local, Eigen::VectorXd &gradient,
                      const size_t &vi1, const size_t &vi2, const size_t &vi3 = -1, const size_t &gammai1 = -1, const size_t &gammai2 = -1);

    template <int Size>
    void fillHessian(const Eigen::Matrix<double, Size, Size> &hessian_local, std::vector<Eigen::Triplet<double>> &hessian_triplets,
                     const size_t &vi1, const size_t &vi2, const size_t &vi3 = -1, const size_t &gammai1 = -1, const size_t &gammai2 = -1);

    void fillMatrix11d(const Eigen::Matrix3d &dd_ec_ec, const Eigen::Matrix3d &dd_ec_ep, const Eigen::Matrix3d &dd_ep_ec, const Eigen::Matrix3d &dd_ep_ep,
                       const Eigen::Vector3d &dd_ec_gamma_c, const Eigen::Vector3d &dd_ec_gamma_p, const Eigen::Vector3d &dd_ep_gamma_c, const Eigen::Vector3d &dd_ep_gamma_p,
                       const double &dd_gamma_c_gamma_c, const double &dd_gamma_c_gamma_p, const double &dd_gamma_p_gamma_c, double &dd_gamma_p_gamma_p,
                       Eigen::Matrix<double, 11, 11> &result);

private:
    // variables
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<double> gammas_;
    // frames
    std::vector<Eigen::Vector3d> reference_frames1_, reference_frames2_;
    std::vector<Eigen::Vector3d> material_frames1_, material_frames2_;
    // vertex-based attributes
    std::vector<double> undeformed_voronoi_lengths_;
    std::vector<Eigen::Vector3d> curvatures_;
    std::vector<Eigen::Vector2d> undeformed_kappas_, kappas_;
    std::vector<Eigen::Vector3d> velocitys_;
    // edge-based attributes
    std::vector<Eigen::Vector3d> undeformed_edges_, edges_;
    std::vector<double> undeformed_gammas_;
    std::vector<double> reference_twists_;
    // constants
    double ks_ = 1000.0, kb_ = 0.00001, kt_ = 0.001; // TODO: tune these parameters
    size_t num_vertices_, num_edges_;
    double mass_ = 1.0;
    Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, -9.8);
    // indices to the DBC
    std::vector<size_t> DBC_vertices_, DBC_gammas_;
};

} // namespace xuan