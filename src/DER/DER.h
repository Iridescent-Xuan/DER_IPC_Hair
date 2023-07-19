#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include "../LinearSolver/EigenSolver.h"

namespace xuan {

class DER {
public:
    DER(const std::vector<Eigen::Vector3d> &vertices, const std::vector<int> &DBC_vertices, const std::vector<int> &DBC_gamma,
        const std::vector<double> &mass, double h = 0.01, Eigen::Vector3d gravity = Eigen::Vector3d(0, -9.8, 0),
        double ks = 1.0, double kb = 1.0, double kt = 1.0)
        : undeformed_vertices_(vertices), vertices_(vertices), DBC_vertices_(DBC_vertices), DBC_gamma_(DBC_gamma),
          mass_(mass), h_(h), gravity_(gravity),
          ks_(ks), kb_(kb), kt_(kt) {
        undeformed_edges_.resize(vertices.size() - 1);
        edges_.resize(vertices.size() - 1);
        tangents_.resize(vertices.size() - 1);
        voronoi_length_.resize(vertices.size(), 0.0);
        gamma_.resize(vertices.size() - 1, 0.0);
        reference_twist_.resize(vertices.size() - 1, 0.0);
        curvature_.resize(vertices.size(), Eigen::Vector3d::Zero());
        undeformed_kappa_.resize(vertices.size(), Eigen::Vector2d::Zero());
        velocity_.resize(vertices.size(), Eigen::Vector3d::Zero());
        x_t_.resize(3 * vertices.size());

        for (int i = 0; i < vertices.size(); ++i) {
            x_t_.segment<3>(3 * i) = vertices[i];
        }

        for (int i = 0; i < vertices.size() - 1; ++i) {
            undeformed_edges_[i] = vertices[i + 1] - vertices[i];
            tangents_[i] = undeformed_edges_[i].normalized();
        }
        edges_ = undeformed_edges_;
        for (int i = 1; i < vertices.size() - 1; ++i) {
            voronoi_length_[i] = (undeformed_edges_[i - 1].norm() + undeformed_edges_[i].norm()) / 2.0;
        }
        updateCurvature();

        { // build the initial Bishop frame
            reference_frame1_.resize(vertices.size() - 1);
            reference_frame2_.resize(vertices.size() - 1);
            material_frame1_.resize(vertices.size() - 1);
            material_frame2_.resize(vertices.size() - 1);

            // initialize a_1^0 and a_2^0 from t^0
            // first choose two arbitrary vectors nonlinear to t^0
            Eigen::Vector3d u, v;
            if (tangents_[0].cross(Eigen::Vector3d::UnitX()).norm() < 1e-8) {
                u = Eigen::Vector3d::UnitY();
                v = Eigen::Vector3d::UnitZ();
            } else if (tangents_[0].cross(Eigen::Vector3d::UnitY()).norm() < 1e-8) {
                u = Eigen::Vector3d::UnitX();
                v = Eigen::Vector3d::UnitZ();
            } else {
                u = Eigen::Vector3d::UnitX();
                v = Eigen::Vector3d::UnitY();
            }
            // then compute the orthogonal frame using the Gram-Schmidt process
            u -= tangents_[0].dot(u) * tangents_[0];
            u.normalize();
            v -= tangents_[0].dot(v) * tangents_[0] + u.dot(v) * u;
            v.normalize();
            assert(fabs(u.dot(v)) < 1e-8 && fabs(u.dot(tangents_[0])) < 1e-8 && fabs(v.dot(tangents_[0]) < 1e-8));
            reference_frame1_[0] = u;
            reference_frame2_[0] = v;
            // initialize the rest of the frames using parallel transport
            for (int i = 1; i < vertices.size() - 1; ++i) {
                reference_frame1_[i] = parallelTransport(reference_frame1_[i - 1], tangents_[i - 1], tangents_[i]);
                reference_frame2_[i] = parallelTransport(reference_frame2_[i - 1], tangents_[i - 1], tangents_[i]);
            }
            updateMaterialFrame();
            updateReferenceTwist();
            undeformed_reference_twist_ = reference_twist_;
            // initialize the undeformed discrete curvature
            for (int i = 1; i < vertices.size() - 1; ++i) {
                undeformed_kappa_[i] = (Eigen::Vector2d(curvature_[i].dot(material_frame2_[i - 1]), -curvature_[i].dot(material_frame1_[i - 1])) + Eigen::Vector2d(curvature_[i].dot(material_frame2_[i]), -curvature_[i].dot(material_frame1_[i]))) / 2.0;
            }
        }
    }

    double computeStretchEnergy();
    void computeStretchGradient(Eigen::VectorXd &grad);
    void computeStretchHessian(Eigen::SparseMatrix<double> &hess);

    double computeBendEnergy();
    void computeBendGradient(Eigen::VectorXd &grad);
    void computeBendHessian(Eigen::SparseMatrix<double> &hess);

    double computeTwistEnergy();
    void computeTwistGradient(Eigen::VectorXd &grad);
    void computeTwistHessian(Eigen::SparseMatrix<double> &hess);

    double computeElasticEnergy() {
        return computeStretchEnergy() + computeBendEnergy() + computeTwistEnergy();
    }

    double computeEnergy(const Eigen::VectorXd &x_gamma);

    bool solve();

    bool applyDBC(const std::vector<Eigen::Vector3d> &DBCvertices, const std::vector<double> &DBCgamma) {
        if ((DBC_vertices_[0] != -1 && DBC_vertices_.size() != DBCvertices.size()) || (DBC_gamma_[0] != -1 && DBC_gamma_.size() != DBCgamma.size())) {
            spdlog::error("The number of DBC vertices or gamma is not correct.");
            return false;
        }
        for (int i = 0; i < DBC_vertices_.size(); ++i) {
            if (DBC_vertices_[i] >= 0 && DBC_vertices_[i] < vertices_.size())
                vertices_[DBC_vertices_[i]] = DBCvertices[i];
            else if (DBC_vertices_[i] != -1)
                spdlog::error("DBC vertex index out of range.");
        }
        for (int i = 0; i < DBC_gamma_.size(); ++i) {
            if (DBC_gamma_[i] >= 0 && DBC_gamma_[i] < gamma_.size())
                gamma_[DBC_gamma_[i]] = DBCgamma[i];
            else if (DBC_gamma_[i] != -1)
                spdlog::error("DBC gamma index out of range.");
        }
        return true;
    }

    void updateReferenceTwist() {
        for (int i = 1; i < edges_.size(); ++i) {
            Eigen::Vector3d space_parallel_transport = parallelTransport(reference_frame1_[i - 1], tangents_[i - 1], tangents_[i]);
            // signed angle from P_{t^{i-1}}^{t^i} a_1^{i-1} to a_1^i
            space_parallel_transport.normalize();
            double cos_angle = space_parallel_transport.dot(reference_frame1_[i].normalized());
            double sin_angle = space_parallel_transport.cross(reference_frame1_[i].normalized()).norm();
            reference_twist_[i] = std::atan2(sin_angle, cos_angle);
        }
    }

    void updateEdges(const std::vector<Eigen::Vector3d> &vertices) {
        for (int i = 0; i < edges_.size(); ++i) {
            Eigen::Vector3d new_edge = vertices[i + 1] - vertices[i];
            Eigen::Vector3d new_tangent = new_edge.normalized();
            // note that we should update the reference frame first
            reference_frame1_[i] = parallelTransport(reference_frame1_[i], tangents_[i], new_tangent);
            reference_frame2_[i] = parallelTransport(reference_frame2_[i], tangents_[i], new_tangent);
            edges_[i] = new_edge;
            tangents_[i] = new_tangent;
        }
    }

    void updateCurvature() {
        for (int i = 1; i < edges_.size(); ++i) {
            curvature_[i] = 2.0 * tangents_[i - 1].cross(tangents_[i]) / (1 + tangents_[i - 1].dot(tangents_[i]));
        }
    }

    void updateMaterialFrame() {
        for (int i = 0; i < edges_.size(); ++i) {
            material_frame1_[i] = rotate(reference_frame1_[i], tangents_[i], gamma_[i]);
            material_frame2_[i] = rotate(reference_frame2_[i], tangents_[i], gamma_[i]);
        }
    }

    void update(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> &gamma) {
        updateEdges(vertices);
        updateCurvature();
        updateMaterialFrame();
        updateReferenceTwist();
        vertices_ = vertices;
        gamma_ = gamma;
    }

    void update(const Eigen::VectorXd &x) {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<double> gamma;
        // check the boundary condition
        bool flag = true;
        for (auto i : DBC_vertices_) {
            if (i >= 0 && i < vertices_.size())
                if (fabs(x[3 * i]) + fabs(x[3 * i + 1]) + fabs(x[3 * i + 2]) > 1e-10)
                    flag = false;
        }
        for (auto i : DBC_gamma_) {
            if (i >= 0 && i < gamma_.size())
                if (fabs(x[x.size() / 3 + i]) > 1e-10)
                    flag = false;
        }
        if (!flag) {
            spdlog::error("The boundary condition is not satisfied.");
            spdlog::error("x = {}", x.transpose());
        }
        assert(flag && "The boundary condition is not satisfied.");

        for (int i = 0; i < vertices_.size(); ++i) {
            vertices.push_back(vertices_[i] + Eigen::Vector3d(x[3 * i], x[3 * i + 1], x[3 * i + 2]));
        }
        for (int i = 0; i < gamma_.size(); ++i) {
            gamma.push_back(gamma_[i] + x[x.size() / 3 + i]);
        }
        update(vertices, gamma);

        // update the velocity
        for (int i = 0; i < vertices.size(); ++i) {
            velocity_[i] = Eigen::Vector3d(x[3 * i], x[3 * i + 1], x[3 * i + 2]) / h_;
        }
    }

    Eigen::Vector3d rotate(const Eigen::Vector3d &v, const Eigen::Vector3d &axis, double angle) {
        Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        return R * v;
    }

    Eigen::Vector3d parallelTransport(const Eigen::Vector3d &target, const Eigen::Vector3d &u, const Eigen::Vector3d &v) {
        Eigen::Vector3d b = u.cross(v);
        Eigen::Vector3d result = target;
        if (b.norm() < 1e-8) {
            return result; // u and v are nearly parallel
        }
        b.normalize();

        Eigen::Vector3d n0 = u.cross(b).normalized();
        Eigen::Vector3d n1 = v.cross(b).normalized();

        result = target.dot(u.normalized()) * v.normalized() + target.dot(n0) * n1 + target.dot(b) * b;
        return result;
    }

    void writeObj(const std::string &filename) {
        std::ofstream out(filename);
        if (!out.is_open()) {
            spdlog::error("Cannot open file {}", filename);
            return;
        }
        for (int i = 0; i < vertices_.size(); ++i) {
            out << "v " << vertices_[i].transpose() << std::endl;
        }
        for (int i = 0; i < vertices_.size() - 1; ++i) {
            out << "l " << i + 1 << " " << i + 2 << std::endl;
        }
        out.close();
    }

    Eigen::MatrixXd symmetricMatrix(const Eigen::MatrixXd &A) {
        return 0.5 * (A + A.transpose());
    }

    Eigen::MatrixXd skewSymmetricVector(const Eigen::Vector3d &v) {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(3, 3);
        result(0, 1) = -v(2);
        result(0, 2) = v(1);
        result(1, 0) = v(2);
        result(1, 2) = -v(0);
        result(2, 0) = -v(1);
        result(2, 1) = v(0);
        return result;
    }

    Eigen::Matrix3d tensorProduct3d(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
        Eigen::Matrix3d result;
        for (int i = 0; i < 3; ++i) {
            result.col(i) = a * b(i);
        }
        return result;
    }

    template <typename Scalar, int size>
    static void makePD(Eigen::Matrix<Scalar, size, size> &symMtr) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            } else {
                // break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }

    void boundaryDBC(Eigen::SparseMatrix<double> &hess);

    void writeGradHessian(const Eigen::VectorXd &grad, const Eigen::SparseMatrix<double> &hess, const std::string &filename = "grad_hessian.txt") {
        std::ofstream out(filename);
        if (!out.is_open()) {
            spdlog::error("Cannot open file {}", filename);
            return;
        }
        out << "gradient = " << std::endl;
        out << grad << std::endl;
        out << "hessian = " << std::endl;
        out << hess << std::endl;
        out.close();
    }

    // helper functions for debugging
    void printGamma() {
        spdlog::info("gamma = ");
        for (auto gamma : gamma_) {
            spdlog::info("{}", gamma);
        }
    }

    void printFrames() {
        spdlog::info("reference frame 1: ");
        for (auto frame : reference_frame1_) {
            spdlog::info("{}", frame.transpose());
        }
        spdlog::info("reference frame 2: ");
        for (auto frame : reference_frame2_) {
            spdlog::info("{}", frame.transpose());
        }
        spdlog::info("material frame 1: ");
        for (auto frame : material_frame1_) {
            spdlog::info("{}", frame.transpose());
        }
        spdlog::info("material frame 2: ");
        for (auto frame : material_frame2_) {
            spdlog::info("{}", frame.transpose());
        }
    }

private:
    std::vector<Eigen::Vector3d> undeformed_vertices_; // 0 ~ n + 1
    std::vector<double> mass_;                         // 0 ~ n + 1
    std::vector<Eigen::Vector3d> velocity_;            // 0 ~ n + 1
    std::vector<Eigen::Vector3d> undeformed_edges_;    // 0 ~ n
    std::vector<double> voronoi_length_;               // 1 ~ n
    std::vector<double> undeformed_reference_twist_;
    std::vector<Eigen::Vector2d> undeformed_kappa_;
    std::vector<Eigen::Vector3d> vertices_;         // updated vertices
    std::vector<double> gamma_;                     // angles for the material frame
    std::vector<Eigen::Vector3d> reference_frame1_; // reference frame a1
    std::vector<Eigen::Vector3d> reference_frame2_; // reference frame a2
    std::vector<Eigen::Vector3d> material_frame1_;  // material frame m1
    std::vector<Eigen::Vector3d> material_frame2_;  // material frame m2
    std::vector<double> reference_twist_;           // reference twist
    std::vector<Eigen::Vector3d> edges_, tangents_;
    std::vector<Eigen::Vector3d> curvature_; // \kappa b; 1 ~ n
    const double ks_, kb_, kt_;              // elastic constants
    double h_;                               // time step size
    Eigen::Vector3d gravity_;
    std::vector<int> DBC_vertices_; // indices of fixed vertices
    std::vector<int> DBC_gamma_;    // indices of fixed gamma
    Eigen::VectorXd x_t_;           // x(t)
};

} // namespace xuan