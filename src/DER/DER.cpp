#include "DER.h"
#include "utils.h"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <fstream>

using namespace Eigen;
using namespace std;

namespace xuan {

DER::DER(const std::vector<Eigen::Vector3d> vertices, const std::vector<double> gammas, const std::vector<size_t> DBC_vertices, const std::vector<size_t> DBC_gammas)
    : vertices_(vertices), gammas_(gammas), DBC_vertices_(DBC_vertices), DBC_gammas_(DBC_gammas) {
    assert(vertices_.size() == gammas_.size() + 1 && "unmatched number of vertices and gammas");
    num_vertices_ = vertices_.size();
    num_edges_ = num_vertices_ - 1;
    // reserve space for vectors
    reference_frames1_.resize(num_edges_);
    reference_frames2_.resize(num_edges_);
    material_frames1_.resize(num_edges_);
    material_frames2_.resize(num_edges_);
    undeformed_voronoi_lengths_.resize(num_vertices_);
    undeformed_kappas_.resize(num_vertices_);
    kappas_.resize(num_vertices_);
    curvatures_.resize(num_vertices_);
    undeformed_edges_.resize(num_edges_);
    edges_.resize(num_edges_);
    undeformed_gammas_.resize(num_edges_);
    reference_twists_.resize(num_edges_);
    velocitys_.resize(num_vertices_, Eigen::Vector3d::Zero());

    // initialize all related attributes
    update(vertices, gammas,
           reference_frames1_, reference_frames2_, material_frames1_, material_frames2_,
           edges_, reference_twists_, curvatures_, kappas_, true);
    // initialize undeformed parameters
    undeformed_edges_ = edges_;
    undeformed_gammas_ = gammas_;
    for (int i = 0; i < num_edges_; ++i) {
        if (undeformed_edges_[i].norm() < 1e-8) {
            spdlog::warn("edge {}: zero length!", i);
        }
    }
    undeformed_kappas_ = kappas_;
    undeformed_voronoi_lengths_[0] = edges_[0].norm();
    undeformed_voronoi_lengths_[num_vertices_ - 1] = edges_[num_vertices_ - 2].norm();
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        undeformed_voronoi_lengths_[i] = (edges_[i].norm() + edges_[i - 1].norm()) / 2.0;
    }
}

double DER::stretchEnergyLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices) {
    assert(validEdgeIndex(index_edge));
    Eigen::Vector3d edge = vertices[index_edge + 1] - vertices[index_edge];
    return ks_ * std::pow(edge.norm() / undeformed_edges_[index_edge].norm() - 1, 2) * undeformed_edges_[index_edge].norm() / 2.0;
}

double DER::stretchEnergy(const std::vector<Eigen::Vector3d> &vertices) {
    assert(vertices.size() == num_vertices_);
    double result = 0.0;
    for (int i = 0; i < num_edges_; ++i) {
        result += stretchEnergyLocal(i, vertices);
    }
    return result;
}

Eigen::Vector<double, 6> DER::stretchGradientLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices) {
    assert(validEdgeIndex(index_edge));
    Eigen::Vector3d edge = vertices[index_edge + 1] - vertices[index_edge];
    Eigen::Vector3d d_edge = ks_ * (edge.norm() / undeformed_edges_[index_edge].norm() - 1) * edge.normalized();
    Eigen::Vector<double, 6> result;
    result.segment<3>(0) = d_edge;
    result.segment<3>(3) = -d_edge;
    return result;
}

void DER::stretchGradient(const std::vector<Eigen::Vector3d> &vertices, Eigen::VectorXd &gradient) {
    assert(vertices.size() == num_vertices_);
    assert(gradient.size() == (3 * num_vertices_ + num_edges_));

    for (int i = 0; i < num_edges_; ++i) {
        Eigen::Vector<double, 6> gradient_local = stretchGradientLocal(i, vertices);
        fillGradient(gradient_local, gradient, i + 1, i);
    }
}

Eigen::Matrix<double, 6, 6> DER::stretchHessianLocal(const size_t &index_edge, const std::vector<Eigen::Vector3d> &vertices) {
    assert(validEdgeIndex(index_edge));
    Eigen::Vector3d edge = vertices[index_edge + 1] - vertices[index_edge];
    Eigen::Matrix3d d_edge_d_edge = ks_ * ((1.0 / undeformed_edges_[index_edge].norm() - 1.0 / edge.norm()) * Eigen::Matrix3d::Identity() + outerProduct(edge, edge) * std::pow(edge.norm(), 3));
    Eigen::Matrix<double, 6, 6> result;
    result.block<3, 3>(0, 0) = d_edge_d_edge;
    result.block<3, 3>(0, 3) = -d_edge_d_edge;
    result.block<3, 3>(3, 0) = -d_edge_d_edge;
    result.block<3, 3>(3, 3) = d_edge_d_edge;
    return result;
}

void DER::stretchHessian(const std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Triplet<double>> &hessian_triplets) {
    assert(vertices.size() == num_vertices_);

    for (int i = 0; i < num_edges_; ++i) {
        Eigen::Matrix<double, 6, 6> hessian_local = stretchHessianLocal(i, vertices);
        makePD(hessian_local);
        fillHessian(hessian_local, hessian_triplets, i + 1, i);
    }
}

double DER::bendEnergyLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas) {
    assert(validVertexIndex(index_vertex));
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        return 0.0;
    }
    return kb_ * (kappas[index_vertex] - undeformed_kappas_[index_vertex]).squaredNorm() / undeformed_voronoi_lengths_[index_vertex] / 2.0;
}

double DER::bendEnergy(const std::vector<Eigen::Vector2d> &kappas) {
    assert(kappas.size() == num_vertices_);
    double result = 0.0;
    for (int i = 0; i < num_vertices_; ++i) {
        result += bendEnergyLocal(i, kappas);
    }
    return result;
}

void DER::kappaGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                             const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                             Eigen::Vector3d &d_kappa1_e_c, Eigen::Vector3d &d_kappa1_e_p, Eigen::Vector3d &d_kappa2_e_c, Eigen::Vector3d &d_kappa2_e_p,
                             double &d_kappa1_gamma_c, double &d_kappa1_gamma_p, double &d_kappa2_gamma_c, double &d_kappa2_gamma_p) {
    assert(validVertexIndex(index_vertex));
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        d_kappa1_e_c.setZero();
        d_kappa1_e_p.setZero();
        d_kappa2_e_c.setZero();
        d_kappa2_e_p.setZero();
        d_kappa1_gamma_c = 0.0;
        d_kappa1_gamma_p = 0.0;
        d_kappa2_gamma_c = 0.0;
        d_kappa2_gamma_p = 0.0;
        return;
    }

    Eigen::Vector3d tangent_i_p = edges[index_vertex - 1].normalized();
    Eigen::Vector3d tangent_i_c = edges[index_vertex].normalized();
    double chi = 1 + tangent_i_p.dot(tangent_i_c);
    Eigen::Vector3d tilde_tangent = (tangent_i_p + tangent_i_c) / chi;
    Eigen::Vector3d tilde_m1 = (material_frames1[index_vertex - 1] + material_frames1[index_vertex]) / chi;
    Eigen::Vector3d tilde_m2 = (material_frames2[index_vertex - 1] + material_frames2[index_vertex]) / chi;

    d_kappa1_e_c = (-kappas[index_vertex][0] * tilde_tangent - tangent_i_p.cross(tilde_m2)) / edges[index_vertex].norm();
    d_kappa1_e_p = (-kappas[index_vertex][0] * tilde_tangent + tangent_i_c.cross(tilde_m2)) / edges[index_vertex - 1].norm();
    d_kappa2_e_c = (-kappas[index_vertex][1] * tilde_tangent - tangent_i_p.cross(tilde_m1)) / edges[index_vertex].norm();
    d_kappa2_e_p = (-kappas[index_vertex][1] * tilde_tangent + tangent_i_c.cross(tilde_m1)) / edges[index_vertex - 1].norm();

    d_kappa1_gamma_c = -material_frames1[index_vertex].dot(curvatures[index_vertex]);
    d_kappa1_gamma_p = -material_frames1[index_vertex - 1].dot(curvatures[index_vertex]);
    d_kappa2_gamma_c = -material_frames2[index_vertex].dot(curvatures[index_vertex]);
    d_kappa2_gamma_p = -material_frames2[index_vertex - 1].dot(curvatures[index_vertex]);
}

Eigen::Vector<double, 11> DER::bendGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                                                 const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2) {
    assert(validVertexIndex(index_vertex));

    Eigen::Vector3d d_kappa1_e_c, d_kappa1_e_p, d_kappa2_e_c, d_kappa2_e_p;
    double d_kappa1_gamma_c, d_kappa1_gamma_p, d_kappa2_gamma_c, d_kappa2_gamma_p;

    Eigen::Vector<double, 11> result;
    result.setZero();

    kappaGradientLocal(index_vertex, kappas, edges, curvatures, material_frames1, material_frames2,
                       d_kappa1_e_c, d_kappa1_e_p, d_kappa2_e_c, d_kappa2_e_p,
                       d_kappa1_gamma_c, d_kappa1_gamma_p, d_kappa2_gamma_c, d_kappa2_gamma_p);

    Eigen::Vector3d d_energy_e_i_p = kb_ / undeformed_voronoi_lengths_[index_vertex] *
                                     ((kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * d_kappa1_e_p +
                                      (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * d_kappa2_e_p);
    Eigen::Vector3d d_energy_e_i_c = kb_ / undeformed_voronoi_lengths_[index_vertex] *
                                     ((kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * d_kappa1_e_c +
                                      (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * d_kappa2_e_c);
    double d_energy_gamma_i_p = kb_ / undeformed_voronoi_lengths_[index_vertex] *
                                ((kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * d_kappa1_gamma_p +
                                 (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * d_kappa2_gamma_p);
    double d_energy_gamma_i_c = kb_ / undeformed_voronoi_lengths_[index_vertex] *
                                ((kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * d_kappa1_gamma_c +
                                 (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * d_kappa2_gamma_c);

    result.segment<3>(0) -= d_energy_e_i_p;
    result.segment<3>(3) -= d_energy_e_i_c;
    result.segment<3>(3) += d_energy_e_i_p;
    result.segment<3>(6) += d_energy_e_i_c;
    result(9) += d_energy_gamma_i_p;
    result(10) += d_energy_gamma_i_c;

    return result;
}

void DER::bendGradient(const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                       const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                       Eigen::VectorXd &gradient) {
    assert(kappas.size() == num_vertices_);
    assert(gradient.size() == (3 * num_vertices_ + num_edges_));

    for (int i = 1; i < num_vertices_ - 1; ++i) {
        Eigen::Vector<double, 11> gradient_local = bendGradientLocal(i, kappas, edges, curvatures, material_frames1, material_frames2);
        fillGradient(gradient_local, gradient, i - 1, i, i + 1, i - 1, i);
    }
}

void DER::kappaHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                            const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                            Eigen::Vector3d &dd_kappa1_ec_gamma_c, Eigen::Vector3d &dd_kappa1_ec_gamma_p, Eigen::Vector3d &dd_kappa1_ep_gamma_c, Eigen::Vector3d &dd_kappa1_ep_gamma_p,
                            Eigen::Vector3d &dd_kappa2_ec_gamma_c, Eigen::Vector3d &dd_kappa2_ec_gamma_p, Eigen::Vector3d &dd_kappa2_ep_gamma_c, Eigen::Vector3d &dd_kappa2_ep_gamma_p,
                            double &dd_kappa1_gamma_c_gamma_c, double &dd_kappa1_gamma_c_gamma_p, double &dd_kappa1_gamma_p_gamma_p,
                            double &dd_kappa2_gamma_c_gamma_c, double &dd_kappa2_gamma_c_gamma_p, double &dd_kappa2_gamma_p_gamma_p) {
    assert(validVertexIndex(index_vertex));
    dd_kappa1_ec_gamma_c.setZero();
    dd_kappa1_ec_gamma_p.setZero();
    dd_kappa1_ep_gamma_c.setZero();
    dd_kappa1_ep_gamma_p.setZero();
    dd_kappa2_ec_gamma_c.setZero();
    dd_kappa2_ec_gamma_p.setZero();
    dd_kappa2_ep_gamma_c.setZero();
    dd_kappa2_ep_gamma_p.setZero();
    dd_kappa1_gamma_c_gamma_c = 0.0;
    dd_kappa1_gamma_c_gamma_p = 0.0;
    dd_kappa1_gamma_p_gamma_p = 0.0;
    dd_kappa2_gamma_c_gamma_c = 0.0;
    dd_kappa2_gamma_c_gamma_p = 0.0;
    dd_kappa2_gamma_p_gamma_p = 0.0;
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        return;
    }

    Eigen::Vector3d tangent_i_p = edges[index_vertex - 1].normalized();
    Eigen::Vector3d tangent_i_c = edges[index_vertex].normalized();
    double chi = 1 + tangent_i_p.dot(tangent_i_c);
    Eigen::Vector3d tilde_tangent = (tangent_i_p + tangent_i_c) / chi;
    Eigen::Vector3d tilde_m1_p = 2 * material_frames1[index_vertex - 1] / chi;
    Eigen::Vector3d tilde_m1_c = 2 * material_frames1[index_vertex] / chi;
    Eigen::Vector3d tilde_m2_p = 2 * material_frames2[index_vertex - 1] / chi;
    Eigen::Vector3d tilde_m2_c = 2 * material_frames2[index_vertex] / chi;

    double kappa1_p = material_frames2[index_vertex - 1].dot(curvatures[index_vertex]);
    double kappa1_c = material_frames2[index_vertex].dot(curvatures[index_vertex]);
    double kappa2_p = -material_frames1[index_vertex - 1].dot(curvatures[index_vertex]);
    double kappa2_c = -material_frames1[index_vertex].dot(curvatures[index_vertex]);

    dd_kappa1_ep_gamma_p += (-kappa2_p * tilde_tangent - tangent_i_c.cross(tilde_m1_p)) / edges[index_vertex - 1].norm();
    dd_kappa1_ec_gamma_p += (-kappa2_p * tilde_tangent + tangent_i_p.cross(tilde_m1_p)) / edges[index_vertex].norm();
    dd_kappa1_gamma_p_gamma_p += -curvatures[index_vertex].dot(material_frames2[index_vertex - 1]);
    dd_kappa2_ep_gamma_p += (kappa1_p * tilde_tangent - tangent_i_c.cross(tilde_m2_p)) / edges[index_vertex - 1].norm();
    dd_kappa2_ec_gamma_p += (kappa1_p * tilde_tangent + tangent_i_p.cross(tilde_m2_p)) / edges[index_vertex].norm();
    dd_kappa2_gamma_p_gamma_p += curvatures[index_vertex].dot(material_frames1[index_vertex - 1]);
    dd_kappa1_ep_gamma_c += (-kappa2_c * tilde_tangent - tangent_i_c.cross(tilde_m1_c)) / edges[index_vertex - 1].norm();
    dd_kappa1_ec_gamma_c += (-kappa2_c * tilde_tangent + tangent_i_p.cross(tilde_m1_c)) / edges[index_vertex].norm();
    dd_kappa1_gamma_c_gamma_c += -curvatures[index_vertex].dot(material_frames2[index_vertex]);
    dd_kappa2_ep_gamma_c += (kappa1_c * tilde_tangent - tangent_i_c.cross(tilde_m2_c)) / edges[index_vertex - 1].norm();
    dd_kappa2_ec_gamma_c += (kappa1_c * tilde_tangent + tangent_i_p.cross(tilde_m2_c)) / edges[index_vertex].norm();
    dd_kappa2_gamma_c_gamma_c += curvatures[index_vertex].dot(material_frames1[index_vertex]);

    dd_kappa1_ec_gamma_c /= 2.0;
    dd_kappa1_ep_gamma_c /= 2.0;
    dd_kappa2_ec_gamma_c /= 2.0;
    dd_kappa2_ep_gamma_c /= 2.0;
    dd_kappa1_ec_gamma_p /= 2.0;
    dd_kappa1_ep_gamma_p /= 2.0;
    dd_kappa2_ec_gamma_p /= 2.0;
    dd_kappa2_ep_gamma_p /= 2.0;
    dd_kappa1_gamma_c_gamma_c /= 2.0;
    dd_kappa1_gamma_c_gamma_p /= 2.0;
    dd_kappa1_gamma_p_gamma_p /= 2.0;
    dd_kappa2_gamma_c_gamma_c /= 2.0;
    dd_kappa2_gamma_c_gamma_p /= 2.0;
    dd_kappa2_gamma_p_gamma_p /= 2.0;
}

Eigen::Matrix<double, 11, 11> DER::bendHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                                                    const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2) {
    assert(validVertexIndex(index_vertex));

    Eigen::Vector3d d_kappa1_e_c, d_kappa1_e_p, d_kappa2_e_c, d_kappa2_e_p;
    double d_kappa1_gamma_c, d_kappa1_gamma_p, d_kappa2_gamma_c, d_kappa2_gamma_p;

    kappaGradientLocal(index_vertex, kappas, edges, curvatures, material_frames1, material_frames2,
                       d_kappa1_e_c, d_kappa1_e_p, d_kappa2_e_c, d_kappa2_e_p,
                       d_kappa1_gamma_c, d_kappa1_gamma_p, d_kappa2_gamma_c, d_kappa2_gamma_p);

    Eigen::Vector3d dd_kappa1_ec_gamma_c, dd_kappa1_ec_gamma_p, dd_kappa1_ep_gamma_c, dd_kappa1_ep_gamma_p;
    Eigen::Vector3d dd_kappa2_ec_gamma_c, dd_kappa2_ec_gamma_p, dd_kappa2_ep_gamma_c, dd_kappa2_ep_gamma_p;
    double dd_kappa1_gamma_c_gamma_c, dd_kappa1_gamma_c_gamma_p, dd_kappa1_gamma_p_gamma_p;
    double dd_kappa2_gamma_c_gamma_c, dd_kappa2_gamma_c_gamma_p, dd_kappa2_gamma_p_gamma_p;
    kappaHessianLocal(index_vertex, edges, curvatures, material_frames1, material_frames2,
                      dd_kappa1_ec_gamma_c, dd_kappa1_ec_gamma_p, dd_kappa1_ep_gamma_c, dd_kappa1_ep_gamma_p,
                      dd_kappa2_ec_gamma_c, dd_kappa2_ec_gamma_p, dd_kappa2_ep_gamma_c, dd_kappa2_ep_gamma_p,
                      dd_kappa1_gamma_c_gamma_c, dd_kappa1_gamma_c_gamma_p, dd_kappa1_gamma_p_gamma_p,
                      dd_kappa2_gamma_c_gamma_c, dd_kappa2_gamma_c_gamma_p, dd_kappa2_gamma_p_gamma_p);

    Eigen::Matrix<double, 11, 11> result;
    result.setZero();

    double coeff = kb_ / undeformed_voronoi_lengths_[index_vertex];
    // Note that we ignore the second derivative of kappa, i.e. dd_kappa_e_e
    Eigen::Matrix3d dd_energy_ep_ep = coeff * (outerProduct(d_kappa1_e_p, d_kappa1_e_p) + outerProduct(d_kappa2_e_p, d_kappa2_e_p));
    Eigen::Matrix3d dd_energy_ep_ec = coeff * (outerProduct(d_kappa1_e_p, d_kappa1_e_c) + outerProduct(d_kappa2_e_p, d_kappa2_e_c));
    Eigen::Matrix3d dd_energy_ec_ep = coeff * (outerProduct(d_kappa1_e_c, d_kappa1_e_p) + outerProduct(d_kappa2_e_c, d_kappa2_e_p));
    Eigen::Matrix3d dd_energy_ec_ec = coeff * (outerProduct(d_kappa1_e_c, d_kappa1_e_c) + outerProduct(d_kappa2_e_c, d_kappa2_e_c));
    Eigen::Vector3d dd_energy_ep_gamma_p = coeff * (d_kappa1_gamma_p * d_kappa1_e_p + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_ep_gamma_p + d_kappa2_gamma_p * d_kappa2_e_p + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_ep_gamma_p);
    Eigen::Vector3d dd_energy_ep_gamma_c = coeff * (d_kappa1_gamma_c * d_kappa1_e_p + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_ep_gamma_c + d_kappa2_gamma_c * d_kappa2_e_p + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_ep_gamma_c);
    Eigen::Vector3d dd_energy_ec_gamma_p = coeff * (d_kappa1_gamma_p * d_kappa1_e_c + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_ec_gamma_p + d_kappa2_gamma_p * d_kappa2_e_c + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_ec_gamma_p);
    Eigen::Vector3d dd_energy_ec_gamma_c = coeff * (d_kappa1_gamma_c * d_kappa1_e_c + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_ec_gamma_c + d_kappa2_gamma_c * d_kappa2_e_c + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_ec_gamma_c);
    double dd_energy_gamma_p_gamma_p = coeff * (d_kappa1_gamma_p * d_kappa1_gamma_p + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_gamma_p_gamma_p + d_kappa2_gamma_p * d_kappa2_gamma_p + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_gamma_p_gamma_p);
    double dd_energy_gamma_p_gamma_c = coeff * (d_kappa1_gamma_c * d_kappa1_gamma_p + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_gamma_c_gamma_p + d_kappa2_gamma_c * d_kappa2_gamma_p + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_gamma_c_gamma_p);
    double dd_energy_gamma_c_gamma_p = dd_energy_gamma_p_gamma_c;
    double dd_energy_gamma_c_gamma_c = coeff * (d_kappa1_gamma_c * d_kappa1_gamma_c + (kappas[index_vertex](0) - undeformed_kappas_[index_vertex](0)) * dd_kappa1_gamma_c_gamma_c + d_kappa2_gamma_c * d_kappa2_gamma_c + (kappas[index_vertex](1) - undeformed_kappas_[index_vertex](1)) * dd_kappa2_gamma_c_gamma_c);

    fillMatrix11d(dd_energy_ec_ec, dd_energy_ec_ep, dd_energy_ep_ec, dd_energy_ep_ep,
                  dd_energy_ec_gamma_c, dd_energy_ec_gamma_p, dd_energy_ep_gamma_c, dd_energy_ep_gamma_p,
                  dd_energy_gamma_c_gamma_c, 0.0, 0.0, dd_energy_gamma_p_gamma_p,
                  result);

    return result;
}

void DER::bendHessian(const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector2d> &kappas,
                      const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                      std::vector<Eigen::Triplet<double>> &hessian_triplets) {
    assert(edges.size() == num_edges_);
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        Eigen::Matrix<double, 11, 11> hessian_local = bendHessianLocal(i, kappas, edges, curvatures, material_frames1, material_frames2);
        makePD(hessian_local);
        fillHessian(hessian_local, hessian_triplets, i - 1, i, i + 1, i - 1, i);
    }
}

double DER::twistEnergyLocal(const size_t &index_vertex, const std::vector<double> &gammas, const std::vector<double> &reference_twists) {
    assert(validVertexIndex(index_vertex));
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        return 0.0;
    }
    double m_i = gammas[index_vertex] - gammas[index_vertex - 1] + reference_twists[index_vertex];
    double m_i_bar = undeformed_gammas_[index_vertex] - undeformed_gammas_[index_vertex - 1]; // FIXME: verify that the undeformed reference twist is zero
    return 0.5 * kt_ / undeformed_voronoi_lengths_[index_vertex] * (m_i - m_i_bar) * (m_i - m_i_bar);
}

double DER::twistEnergy(const std::vector<double> &gammas, const std::vector<double> &reference_twists) {
    assert(gammas.size() == num_edges_ && reference_twists.size() == num_edges_);
    double result = 0.0;
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        result += twistEnergyLocal(i, gammas, reference_twists);
    }
    return result;
}

void DER::miGradientLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges,
                          Eigen::Vector3d &d_mi_e_c, Eigen::Vector3d &d_mi_e_p, double &d_mi_gamma_c, double &d_mi_gamma_p) {
    assert(validVertexIndex(index_vertex));
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        d_mi_e_c.setZero();
        d_mi_e_p.setZero();
        d_mi_gamma_c = 0.0;
        d_mi_gamma_p = 0.0;
        return;
    }

    d_mi_e_c = 0.5 * curvatures[index_vertex] / edges[index_vertex].norm();
    d_mi_e_p = 0.5 * curvatures[index_vertex] / edges[index_vertex - 1].norm();
    d_mi_gamma_c = 1.0;
    d_mi_gamma_p = -1.0;
}

Eigen::Vector<double, 11> DER::twistGradientLocal(const size_t &index_vertex, const std::vector<double> gammas, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges, const std::vector<double> &reference_twists) {
    assert(validVertexIndex(index_vertex));
    Eigen::Vector<double, 11> result;
    result.setZero();
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        return result;
    }

    Eigen::Vector3d d_mi_e_c, d_mi_e_p;
    double d_mi_gamma_c, d_mi_gamma_p;
    miGradientLocal(index_vertex, curvatures, edges, d_mi_e_c, d_mi_e_p, d_mi_gamma_c, d_mi_gamma_p);

    double m_i = gammas[index_vertex] - gammas[index_vertex - 1] + reference_twists[index_vertex];
    double m_i_bar = undeformed_gammas_[index_vertex] - undeformed_gammas_[index_vertex - 1];
    Eigen::Vector3d d_energy_e_i_c = kt_ / undeformed_voronoi_lengths_[index_vertex] * (m_i - m_i_bar) * d_mi_e_c;
    Eigen::Vector3d d_energy_e_i_p = kt_ / undeformed_voronoi_lengths_[index_vertex] * (m_i - m_i_bar) * d_mi_e_p;
    double d_energy_gamma_i_c = kt_ / undeformed_voronoi_lengths_[index_vertex] * (m_i - m_i_bar) * d_mi_gamma_c;
    double d_energy_gamma_i_p = kt_ / undeformed_voronoi_lengths_[index_vertex] * (m_i - m_i_bar) * d_mi_gamma_p;

    result.segment<3>(0) -= d_energy_e_i_p;
    result.segment<3>(3) -= d_energy_e_i_c;
    result.segment<3>(3) += d_energy_e_i_p;
    result.segment<3>(6) += d_energy_e_i_c;
    result(9) += d_energy_gamma_i_p;
    result(10) += d_energy_gamma_i_c;

    return result;
}

void DER::twistGradient(const std::vector<double> &gammas, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges, const std::vector<double> &reference_twists, Eigen::VectorXd &gradient) {
    assert(curvatures.size() == num_vertices_ && edges.size() == num_edges_ && reference_twists.size() == num_edges_);
    assert(gradient.size() == (3 * num_vertices_ + num_edges_));

    for (int i = 1; i < num_vertices_ - 1; ++i) {
        Eigen::Vector<double, 11> gradient_local = twistGradientLocal(i, gammas, curvatures, edges, reference_twists);
        fillGradient(gradient_local, gradient, i - 1, i, i + 1, i - 1, i);
    }
}

Eigen::Matrix<double, 11, 11> DER::twistHessianLocal(const size_t &index_vertex, const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges) {
    assert(validVertexIndex(index_vertex));
    Eigen::Matrix<double, 11, 11> result;
    result.setZero();
    if (index_vertex == 0 || index_vertex == num_vertices_ - 1) {
        return result;
    }

    Eigen::Vector3d d_mi_e_c, d_mi_e_p;
    double d_mi_gamma_c, d_mi_gamma_p;
    miGradientLocal(index_vertex, curvatures, edges, d_mi_e_c, d_mi_e_p, d_mi_gamma_c, d_mi_gamma_p);

    double coeff = kt_ / undeformed_voronoi_lengths_[index_vertex];
    // Note that we ignore the second derivative of m_i, i.e. dd_mi_e_e
    Eigen::Matrix3d dd_energy_e_c_e_c = coeff * outerProduct(d_mi_e_c, d_mi_e_c);
    Eigen::Matrix3d dd_energy_e_c_e_p = coeff * outerProduct(d_mi_e_c, d_mi_e_p);
    Eigen::Matrix3d dd_energy_e_p_e_c = coeff * outerProduct(d_mi_e_p, d_mi_e_c);
    Eigen::Matrix3d dd_energy_e_p_e_p = coeff * outerProduct(d_mi_e_p, d_mi_e_p);
    Eigen::Vector3d dd_energy_e_c_gamma_c = coeff * (d_mi_gamma_c * d_mi_e_c);
    Eigen::Vector3d dd_energy_e_c_gamma_p = coeff * (d_mi_gamma_p * d_mi_e_c);
    Eigen::Vector3d dd_energy_e_p_gamma_c = coeff * (d_mi_gamma_c * d_mi_e_p);
    Eigen::Vector3d dd_energy_e_p_gamma_p = coeff * (d_mi_gamma_p * d_mi_e_p);
    double dd_energy_gamma_c_gamma_c = coeff * (d_mi_gamma_c * d_mi_gamma_c);
    double dd_energy_gamma_c_gamma_p = coeff * (d_mi_gamma_p * d_mi_gamma_c);
    double dd_energy_gamma_p_gamma_c = coeff * (d_mi_gamma_c * d_mi_gamma_p);
    double dd_energy_gamma_p_gamma_p = coeff * (d_mi_gamma_p * d_mi_gamma_p);

    fillMatrix11d(dd_energy_e_c_e_c, dd_energy_e_c_e_p, dd_energy_e_p_e_c, dd_energy_e_p_e_p,
                  dd_energy_e_c_gamma_c, dd_energy_e_c_gamma_p, dd_energy_e_p_gamma_c, dd_energy_e_p_gamma_p,
                  dd_energy_gamma_c_gamma_c, dd_energy_gamma_c_gamma_p, dd_energy_gamma_p_gamma_c, dd_energy_gamma_p_gamma_p,
                  result);

    return result;
}

void DER::twistHessian(const std::vector<Eigen::Vector3d> &curvatures, const std::vector<Eigen::Vector3d> &edges,
                       std::vector<Eigen::Triplet<double>> &hessian_triplets) {
    assert(curvatures.size() == num_vertices_ && edges.size() == num_edges_);
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        Eigen::Matrix<double, 11, 11> hessian_local = twistHessianLocal(i, curvatures, edges);
        makePD(hessian_local);
        fillHessian(hessian_local, hessian_triplets, i - 1, i, i + 1, i - 1, i);
    }
}

double DER::elasticEnergy(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector2d> &kappas, const std::vector<double> &gammas, const std::vector<double> &reference_twists) {
    assert(vertices.size() == num_vertices_ && kappas.size() == num_vertices_ && gammas.size() == num_edges_ && reference_twists.size() == num_edges_);
    return stretchEnergy(vertices) + bendEnergy(kappas) + twistEnergy(gammas, reference_twists);
}

void DER::elasticGradient(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> &gammas, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                          const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2, const std::vector<double> &reference_twists,
                          Eigen::VectorXd &gradient) {
    assert(vertices.size() == num_vertices_ && gammas.size() == num_edges_ && kappas.size() == num_vertices_);
    if (gradient.size() == 0) {
        gradient.resize(3 * num_vertices_ + num_edges_);
        gradient.setZero();
    }
    assert(gradient.size() == 3 * num_vertices_ + num_edges_);

    stretchGradient(vertices, gradient);
    bendGradient(kappas, edges, curvatures, material_frames1, material_frames2, gradient);
    twistGradient(gammas, curvatures, edges, reference_twists, gradient);
}

void DER::elasticHessian(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector2d> &kappas, const std::vector<Eigen::Vector3d> &edges, const std::vector<Eigen::Vector3d> &curvatures,
                         const std::vector<Eigen::Vector3d> &material_frames1, const std::vector<Eigen::Vector3d> &material_frames2,
                         Eigen::SparseMatrix<double> &hessian) {
    assert(vertices.size() == num_vertices_ && curvatures.size() == num_vertices_);
    if (hessian.size() == 0) {
        hessian.resize(3 * num_vertices_ + num_edges_, 3 * num_vertices_ + num_edges_);
    }
    assert(hessian.rows() == 3 * num_vertices_ + num_edges_ && hessian.cols() == 3 * num_vertices_ + num_edges_);
    std::vector<Eigen::Triplet<double>> hessian_triplets;
    stretchHessian(vertices, hessian_triplets);
    bendHessian(edges, curvatures, kappas, material_frames1, material_frames2, hessian_triplets);
    twistHessian(curvatures, edges, hessian_triplets);
    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
}

void DER::elasticAttributes(const Eigen::VectorXd &x,
                            double &elastic_energy, Eigen::VectorXd &elastic_gradient, Eigen::SparseMatrix<double> &elastic_hessian) {
    assert(x.size() == 3 * num_vertices_ + num_edges_);
    std::vector<Eigen::Vector3d> vertices(num_vertices_);
    std::vector<double> gammas(num_edges_);
    for (int i = 0; i < num_vertices_; ++i) {
        vertices[i] = x.segment<3>(3 * i);
    }
    for (int i = 0; i < num_edges_; ++i) {
        gammas[i] = x(3 * num_vertices_ + i);
    }

    std::vector<Eigen::Vector3d> reference_frames1, reference_frames2;
    std::vector<Eigen::Vector3d> material_frames1, material_frames2;
    std::vector<Eigen::Vector3d> edges;
    std::vector<Eigen::Vector3d> curvatures;
    std::vector<Eigen::Vector2d> kappas;
    std::vector<double> reference_twists;
    update(vertices, gammas, reference_frames1, reference_frames2, material_frames1, material_frames2, edges, reference_twists, curvatures, kappas, false);

    elastic_energy = elasticEnergy(vertices, kappas, gammas, reference_twists);
    elasticGradient(vertices, gammas, kappas, edges, curvatures, material_frames1, material_frames2, reference_twists, elastic_gradient);
    elasticHessian(vertices, kappas, edges, curvatures, material_frames1, material_frames2, elastic_hessian);
}

double DER::kineticEnergy(const std::vector<Eigen::Vector3d> &vertices, double h) {
    assert(vertices.size() == num_vertices_);
    double energy = 0.0;
    for (int i = 0; i < num_vertices_; ++i) {
        Eigen::Vector3d x_hat = vertices_[i] + h * velocitys_[i] + h * h * gravity_; // the only external force is gravity
        energy += 0.5 * density_ * undeformed_voronoi_lengths_[i] * (vertices[i] - x_hat).squaredNorm();
    }
    return energy;
}

void DER::kineticGradient(const std::vector<Eigen::Vector3d> &vertices, double h,
                          Eigen::VectorXd &gradient) {
    assert(vertices.size() == num_vertices_);
    if (gradient.size() == 0) {
        gradient.resize(3 * num_vertices_ + num_edges_);
        gradient.setZero();
    }
    assert(gradient.size() == 3 * num_vertices_ + num_edges_);
    for (int i = 0; i < num_vertices_; ++i) {
        Eigen::Vector3d x_hat = vertices_[i] + h * velocitys_[i] + h * h * gravity_;
        gradient.segment<3>(3 * i) += density_ * undeformed_voronoi_lengths_[i] * (vertices[i] - x_hat);
    }
}

void DER::kineticHessian(Eigen::SparseMatrix<double> &hessian) {
    std::vector<Eigen::Triplet<double>> hessian_triplets;
    for (int i = 0; i < num_vertices_; ++i) {
        for (int j = 0; j < 3; ++j) {
            hessian_triplets.emplace_back(3 * i + j, 3 * i + j, density_ * undeformed_voronoi_lengths_[i]);
        }
    }
    if (hessian.size() == 0) {
        hessian.resize(3 * num_vertices_ + num_edges_, 3 * num_vertices_ + num_edges_);
    }
    assert(hessian.rows() == 3 * num_vertices_ + num_edges_ && hessian.cols() == 3 * num_vertices_ + num_edges_);
    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
}

void DER::kineticAttributes(const Eigen::VectorXd &x, double h,
                            double &kinetic_energy, Eigen::VectorXd &kinetic_gradient, Eigen::SparseMatrix<double> &kinetic_hessian) {
    assert(x.size() == 3 * num_vertices_ + num_edges_);
    std::vector<Eigen::Vector3d> vertices(num_vertices_);
    for (int i = 0; i < num_vertices_; ++i) {
        vertices[i] = x.segment<3>(3 * i);
    }
    kinetic_energy = kineticEnergy(vertices, h);
    kineticGradient(vertices, h, kinetic_gradient);
    kineticHessian(kinetic_hessian);
}

void DER::incrementalPotential(const Eigen::VectorXd &x, double h,
                               double &energy, Eigen::VectorXd &gradient, Eigen::SparseMatrix<double> &hessian) {
    assert(x.size() == 3 * num_vertices_ + num_edges_);

    double elastic_energy, kinetic_energy;
    Eigen::VectorXd elastic_gradient, kinetic_gradient;
    Eigen::SparseMatrix<double> elastic_hessian, kinetic_hessian;

    elasticAttributes(x, elastic_energy, elastic_gradient, elastic_hessian);
    kineticAttributes(x, h, kinetic_energy, kinetic_gradient, kinetic_hessian);

    energy = h * h * elastic_energy + kinetic_energy;
    gradient = h * h * elastic_gradient + kinetic_gradient;
    hessian = h * h * elastic_hessian + kinetic_hessian;

    handleDBC(gradient, hessian);
}

void DER::deepUpdate(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> gammas, double h) {
    assert(vertices.size() == num_vertices_ && gammas.size() == num_edges_);

    update(vertices, gammas,
           reference_frames1_, reference_frames2_,
           material_frames1_, material_frames2_,
           edges_, reference_twists_, curvatures_, kappas_, false);

    for (int i = 0; i < num_vertices_; ++i) {
        velocitys_[i] = (vertices[i] - vertices_[i]) / h;
    }
    vertices_ = vertices;
    gammas_ = gammas;
}

void DER::deepUpdate(const Eigen::VectorXd &x, double h) {
    assert(x.size() == 3 * num_vertices_ + num_edges_);
    std::vector<Eigen::Vector3d> vertices(num_vertices_);
    std::vector<double> gammas(num_edges_);
    for (int i = 0; i < num_vertices_; ++i) {
        vertices[i] = x.segment<3>(3 * i);
    }
    for (int i = 0; i < num_edges_; ++i) {
        gammas[i] = x(3 * num_vertices_ + i);
    }
    deepUpdate(vertices, gammas, h);
}

void DER::update(const std::vector<Eigen::Vector3d> &vertices, const std::vector<double> gammas,
                 std::vector<Eigen::Vector3d> &reference_frames1, std::vector<Eigen::Vector3d> &reference_frames2,
                 std::vector<Eigen::Vector3d> &material_frames1, std::vector<Eigen::Vector3d> &material_frames2,
                 std::vector<Eigen::Vector3d> &edges, std::vector<double> &reference_twists,
                 std::vector<Eigen::Vector3d> &curvatures, std::vector<Eigen::Vector2d> &kappas,
                 bool constructor) {
    assert(vertices.size() == num_vertices_ && gammas.size() == num_edges_);

    edges.resize(num_edges_);
    std::vector<Eigen::Vector3d> tangents(num_edges_);
    for (int i = 0; i < num_edges_; ++i) {
        edges[i] = vertices[i + 1] - vertices[i];
        tangents[i] = edges[i].normalized();
    }

    reference_frames1.resize(num_edges_);
    reference_frames2.resize(num_edges_);
    if (constructor) { // space-parallel transport; init only once at t = 0
        // first choose two arbitrary vectors nonlinear to t^0
        Eigen::Vector3d u, v;
        if (tangents[0].cross(Eigen::Vector3d::UnitX()).norm() < 1e-8) {
            u = Eigen::Vector3d::UnitY();
            v = Eigen::Vector3d::UnitZ();
        } else if (tangents[0].cross(Eigen::Vector3d::UnitY()).norm() < 1e-8) {
            u = Eigen::Vector3d::UnitX();
            v = Eigen::Vector3d::UnitZ();
        } else {
            u = Eigen::Vector3d::UnitX();
            v = Eigen::Vector3d::UnitY();
        }
        // then compute the orthogonal frame using the Gram-Schmidt process
        u -= tangents[0].dot(u) * tangents[0];
        u.normalize();
        v -= tangents[0].dot(v) * tangents[0] + u.dot(v) * u;
        v.normalize();
        assert(fabs(u.dot(v)) < 1e-8 && fabs(u.dot(tangents[0])) < 1e-8 && fabs(v.dot(tangents[0]) < 1e-8));
        reference_frames1[0] = u;
        reference_frames2[0] = v;
        // initialize the rest of the frames using parallel transport
        for (int i = 1; i < num_edges_; ++i) {
            reference_frames1[i] = parallelTransport(reference_frames1[i - 1], tangents[i - 1], tangents[i]);
            reference_frames2[i] = parallelTransport(reference_frames2[i - 1], tangents[i - 1], tangents[i]);
        }
    } else { // time-parallel transport; from t to t+1
        std::vector<Eigen::Vector3d> reference_frames1_prev = reference_frames1_;
        std::vector<Eigen::Vector3d> reference_frames2_prev = reference_frames2_;
        for (int i = 0; i < num_edges_; ++i) {
            reference_frames1[i] = parallelTransport(reference_frames1_prev[i], edges_[i].normalized(), tangents[i]);
            reference_frames2[i] = parallelTransport(reference_frames2_prev[i], edges_[i].normalized(), tangents[i]);
        }
    }

    reference_twists.resize(num_edges_);
    for (int i = 1; i < num_edges_; ++i) {
        Eigen::Vector3d space_parallel_transport = parallelTransport(reference_frames1[i - 1], tangents[i - 1], tangents[i]);
        // signed angle from P_{t^{i-1}}^{t^i} a_1^{i-1} to a_1^i
        space_parallel_transport.normalize();
        double cos_angle = space_parallel_transport.dot(reference_frames1[i].normalized());
        double sin_angle = space_parallel_transport.cross(reference_frames1[i].normalized()).norm();
        reference_twists[i] = std::atan2(sin_angle, cos_angle);
    }

    material_frames1.resize(num_edges_);
    material_frames2.resize(num_edges_);
    for (int i = 0; i < num_edges_; ++i) {
        material_frames1[i] = rotate(reference_frames1[i], tangents[i], gammas[i]);
        material_frames2[i] = rotate(reference_frames2[i], tangents[i], gammas[i]);
    }

    curvatures.resize(num_vertices_);
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        curvatures[i] = 2.0 * tangents[i - 1].cross(tangents[i]) / (1 + tangents[i - 1].dot(tangents[i]));
    }

    kappas.resize(num_vertices_);
    for (int i = 1; i < num_vertices_ - 1; ++i) {
        kappas[i] = (Eigen::Vector2d(curvatures[i].dot(material_frames2[i - 1]), -curvatures[i].dot(material_frames1[i - 1])) + Eigen::Vector2d(curvatures[i].dot(material_frames2[i]), -curvatures[i].dot(material_frames1[i]))) / 2.0;
    }
}

void DER::handleDBC(Eigen::VectorXd &gradient, Eigen::SparseMatrix<double> &hessian) {
    assert(gradient.size() == 3 * num_vertices_ + num_edges_);
    assert(hessian.rows() == 3 * num_vertices_ + num_edges_ && hessian.cols() == 3 * num_vertices_ + num_edges_);

    for (auto v : DBC_vertices_) {
        if (validVertexIndex(v)) {
            gradient.segment<3>(3 * v).setZero();
            for (int i = 0; i < 3; ++i) {
                hessian.row(3 * v + i) *= 0.0;
                hessian.col(3 * v + i) *= 0.0;
                hessian.coeffRef(3 * v + i, 3 * v + i) = 1.0;
            }
        }
    }

    for (auto gamma : DBC_gammas_) {
        if (validEdgeIndex(gamma)) {
            gradient(3 * num_vertices_ + gamma) = 0.0;
            hessian.row(3 * num_vertices_ + gamma) *= 0.0;
            hessian.col(3 * num_vertices_ + gamma) *= 0.0;
            hessian.coeffRef(3 * num_vertices_ + gamma, 3 * num_vertices_ + gamma) = 1.0;
        }
    }
}

Eigen::Vector3d DER::parallelTransport(const Eigen::Vector3d &target, const Eigen::Vector3d &u, const Eigen::Vector3d &v) {
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

Eigen::Vector3d DER::rotate(const Eigen::Vector3d &v, const Eigen::Vector3d &axis, double angle) {
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    return R * v;
}

void DER::fillGradient(const Eigen::VectorXd &gradient_local, Eigen::VectorXd &gradient,
                       const size_t &vi1, const size_t &vi2, const size_t &vi3, const size_t &gammai1, const size_t &gammai2) {
    assert(gradient_local.rows() == 6 || gradient_local.rows() == 11);
    assert(gradient.size() == 3 * num_vertices_ + num_edges_);
    assert(validVertexIndex(vi1) && validVertexIndex(vi2));
    gradient.segment<3>(3 * vi1) += gradient_local.segment<3>(0);
    gradient.segment<3>(3 * vi2) += gradient_local.segment<3>(3);
    if (gradient_local.rows() == 11) {
        assert(validVertexIndex(vi3) && validEdgeIndex(gammai1) && validEdgeIndex(gammai2));
        gradient.segment<3>(3 * vi3) += gradient_local.segment<3>(6);
        gradient(3 * num_vertices_ + gammai1) += gradient_local(9);
        gradient(3 * num_vertices_ + gammai2) += gradient_local(10);
    }
}

template <int Size>
void DER::fillHessian(const Eigen::Matrix<double, Size, Size> &hessian_local, std::vector<Eigen::Triplet<double>> &hessian_triplets,
                      const size_t &vi1, const size_t &vi2, const size_t &vi3, const size_t &gammai1, const size_t &gammai2) {
    assert(Size == 6 || Size == 11);
    if (Size == 6) {
        assert(validVertexIndex(vi1) && validVertexIndex(vi2));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                hessian_triplets.emplace_back(3 * vi1 + i, 3 * vi1 + j, hessian_local(i, j));
                hessian_triplets.emplace_back(3 * vi1 + i, 3 * vi2 + j, hessian_local(i, j + 3));
                hessian_triplets.emplace_back(3 * vi2 + i, 3 * vi1 + j, hessian_local(i + 3, j));
                hessian_triplets.emplace_back(3 * vi2 + i, 3 * vi2 + j, hessian_local(i + 3, j + 3));
            }
        }
    } else if (Size == 11) {
        assert(validVertexIndex(vi1) && validVertexIndex(vi2) && validVertexIndex(vi3) && validEdgeIndex(gammai1) && validEdgeIndex(gammai2));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                hessian_triplets.emplace_back(3 * vi1 + i, 3 * vi1 + j, hessian_local(i, j));
                hessian_triplets.emplace_back(3 * vi1 + i, 3 * vi2 + j, hessian_local(i, j + 3));
                hessian_triplets.emplace_back(3 * vi1 + i, 3 * vi3 + j, hessian_local(i, j + 6));
                hessian_triplets.emplace_back(3 * vi2 + i, 3 * vi1 + j, hessian_local(i + 3, j));
                hessian_triplets.emplace_back(3 * vi2 + i, 3 * vi2 + j, hessian_local(i + 3, j + 3));
                hessian_triplets.emplace_back(3 * vi2 + i, 3 * vi3 + j, hessian_local(i + 3, j + 6));
                hessian_triplets.emplace_back(3 * vi3 + i, 3 * vi1 + j, hessian_local(i + 6, j));
                hessian_triplets.emplace_back(3 * vi3 + i, 3 * vi2 + j, hessian_local(i + 6, j + 3));
                hessian_triplets.emplace_back(3 * vi3 + i, 3 * vi3 + j, hessian_local(i + 6, j + 6));
            }
            hessian_triplets.emplace_back(3 * vi1 + i, 3 * num_vertices_ + gammai1, hessian_local(i, 9));
            hessian_triplets.emplace_back(3 * vi1 + i, 3 * num_vertices_ + gammai2, hessian_local(i, 10));
            hessian_triplets.emplace_back(3 * vi2 + i, 3 * num_vertices_ + gammai1, hessian_local(i + 3, 9));
            hessian_triplets.emplace_back(3 * vi2 + i, 3 * num_vertices_ + gammai2, hessian_local(i + 3, 10));
            hessian_triplets.emplace_back(3 * vi3 + i, 3 * num_vertices_ + gammai1, hessian_local(i + 6, 9));
            hessian_triplets.emplace_back(3 * vi3 + i, 3 * num_vertices_ + gammai2, hessian_local(i + 6, 10));
        }
        for (int j = 0; j < 3; ++j) {
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai1, 3 * vi1 + j, hessian_local(9, j));
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai1, 3 * vi2 + j, hessian_local(9, j + 3));
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai1, 3 * vi3 + j, hessian_local(9, j + 6));
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai2, 3 * vi1 + j, hessian_local(10, j));
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai2, 3 * vi2 + j, hessian_local(10, j + 3));
            hessian_triplets.emplace_back(3 * num_vertices_ + gammai2, 3 * vi3 + j, hessian_local(10, j + 6));
        }
        hessian_triplets.emplace_back(3 * num_vertices_ + gammai1, 3 * num_vertices_ + gammai1, hessian_local(9, 9));
        hessian_triplets.emplace_back(3 * num_vertices_ + gammai1, 3 * num_vertices_ + gammai2, hessian_local(9, 10));
        hessian_triplets.emplace_back(3 * num_vertices_ + gammai2, 3 * num_vertices_ + gammai1, hessian_local(10, 9));
        hessian_triplets.emplace_back(3 * num_vertices_ + gammai2, 3 * num_vertices_ + gammai2, hessian_local(10, 10));
    }
}

void DER::fillMatrix11d(const Eigen::Matrix3d &dd_ec_ec, const Eigen::Matrix3d &dd_ec_ep, const Eigen::Matrix3d &dd_ep_ec, const Eigen::Matrix3d &dd_ep_ep,
                        const Eigen::Vector3d &dd_ec_gamma_c, const Eigen::Vector3d &dd_ec_gamma_p, const Eigen::Vector3d &dd_ep_gamma_c, const Eigen::Vector3d &dd_ep_gamma_p,
                        const double &dd_gamma_c_gamma_c, const double &dd_gamma_c_gamma_p, const double &dd_gamma_p_gamma_c, double &dd_gamma_p_gamma_p,
                        Eigen::Matrix<double, 11, 11> &result) {
    result.setZero();
    result.block<3, 3>(0, 0) += dd_ep_ep;
    result.block<3, 3>(3, 3) += dd_ep_ep;
    result.block<3, 3>(0, 3) -= dd_ep_ep;
    result.block<3, 3>(3, 0) -= dd_ep_ep;
    result.block<3, 3>(3, 6) += dd_ep_ec;
    result.block<3, 3>(0, 3) += dd_ep_ec;
    result.block<3, 3>(3, 3) -= dd_ep_ec;
    result.block<3, 3>(0, 6) += dd_ep_ec;
    result.block<3, 3>(6, 3) += dd_ec_ep;
    result.block<3, 3>(3, 0) += dd_ec_ep;
    result.block<3, 3>(6, 0) -= dd_ec_ep;
    result.block<3, 3>(3, 3) -= dd_ec_ep;
    result.block<3, 3>(6, 6) += dd_ec_ec;
    result.block<3, 3>(3, 3) += dd_ec_ec;
    result.block<3, 3>(6, 3) -= dd_ec_ec;
    result.block<3, 3>(3, 6) -= dd_ec_ec;
    result.block<3, 1>(0, 9) -= dd_ep_gamma_p;
    result.block<3, 1>(3, 9) += dd_ep_gamma_p;
    result.block<1, 3>(9, 0) -= dd_ep_gamma_p.transpose();
    result.block<1, 3>(9, 3) += dd_ep_gamma_p.transpose();
    result.block<3, 1>(0, 10) -= dd_ep_gamma_c;
    result.block<3, 1>(3, 10) += dd_ep_gamma_c;
    result.block<1, 3>(10, 0) -= dd_ep_gamma_c.transpose();
    result.block<1, 3>(10, 3) += dd_ep_gamma_c.transpose();
    result.block<3, 1>(3, 9) -= dd_ec_gamma_p;
    result.block<3, 1>(6, 9) += dd_ec_gamma_p;
    result.block<1, 3>(9, 3) -= dd_ec_gamma_p.transpose();
    result.block<1, 3>(9, 6) += dd_ec_gamma_p.transpose();
    result.block<3, 1>(3, 10) -= dd_ec_gamma_c;
    result.block<3, 1>(6, 10) += dd_ec_gamma_c;
    result.block<1, 3>(10, 3) -= dd_ec_gamma_c.transpose();
    result.block<1, 3>(10, 6) += dd_ec_gamma_c.transpose();
    result(9, 9) += dd_gamma_p_gamma_p;
    result(10, 10) += dd_gamma_c_gamma_c;
    result(9, 10) += dd_gamma_p_gamma_c;
    result(10, 9) += dd_gamma_c_gamma_p;
}

void DER::writeOBJ(const std::string &filename, size_t offset) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        spdlog::warn("DER::writeOBJ: cannot open file {}", filename);
        return;
    }

    for (size_t i = 0; i < num_vertices_; ++i) {
        out << "v " << vertices_[i].x() << " " << vertices_[i].y() << " " << vertices_[i].z() << std::endl;
    }
    out << "l";
    for (size_t i = 0; i < num_vertices_; ++i) {
        out << " " << i + 1 + offset;
    }
    out << std::endl;
    out.close();
    spdlog::debug("DER::writeOBJ: write to file {}", filename);
}

void DER::writeOBJ(std::ofstream &out, size_t offset) const {
    if (!out.is_open()) {
        spdlog::warn("DER::writeOBJ: cannot open file");
        return;
    }

    for (size_t i = 0; i < num_vertices_; ++i) {
        out << "v " << vertices_[i].x() << " " << vertices_[i].y() << " " << vertices_[i].z() << std::endl;
    }
    out << "l";
    for (size_t i = 0; i < num_vertices_; ++i) {
        out << " " << i + 1 + offset;
    }
    out << std::endl;
}

Eigen::Vector3d DER::bboxSize() const {
    double min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = std::numeric_limits<double>::max();
    max_x = max_y = max_z = std::numeric_limits<double>::lowest();

    for (auto v : vertices_) {
        if (v.x() < min_x)
            min_x = v.x();
        if (v.y() < min_y)
            min_y = v.y();
        if (v.z() < min_z)
            min_z = v.z();
        if (v.x() > max_x)
            max_x = v.x();
        if (v.y() > max_y)
            max_y = v.y();
        if (v.z() > max_z)
            max_z = v.z();
    }

    return Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);
}

bool DER::violateDBC(const std::vector<Eigen::Vector3d> &vertices_t, const std::vector<double> &gammas_t,
                     const std::vector<Eigen::Vector3d> &vertices_tt, const std::vector<double> &gammas_tt) const {
    assert(vertices_t.size() == num_vertices_ && gammas_t.size() == num_edges_);
    assert(vertices_tt.size() == num_vertices_ && gammas_tt.size() == num_edges_);

    bool violate = false;
    for (auto v : DBC_vertices_) {
        if (validVertexIndex(v)) {
            if ((vertices_t[v] - vertices_tt[v]).norm() > 1e-8) {
                spdlog::warn("DBC vertex {} is not fixed!", v);
                violate = true;
            }
        }
    }
    for (auto i : DBC_gammas_) {
        if (validEdgeIndex(i)) {
            if (fabs(gammas_t[i] - gammas_tt[i]) > 1e-8) {
                spdlog::warn("DBC gamma {} is not fixed!", i);
                violate = true;
            }
        }
    }
    return violate;
}

bool DER::violateDBC(const Eigen::VectorXd &x_t, const Eigen::VectorXd &x_tt) const {
    assert(x_t.size() == 3 * num_vertices_ + num_edges_);
    assert(x_tt.size() == 3 * num_vertices_ + num_edges_);

    std::vector<Eigen::Vector3d> vertices_t(num_vertices_);
    std::vector<double> gammas_t(num_edges_);
    std::vector<Eigen::Vector3d> vertices_tt(num_vertices_);
    std::vector<double> gammas_tt(num_edges_);
    for (int i = 0; i < num_vertices_; ++i) {
        vertices_t[i] = x_t.segment<3>(3 * i);
        vertices_tt[i] = x_tt.segment<3>(3 * i);
    }
    for (int i = 0; i < num_edges_; ++i) {
        gammas_t[i] = x_t(3 * num_vertices_ + i);
        gammas_tt[i] = x_tt(3 * num_vertices_ + i);
    }
    return violateDBC(vertices_t, gammas_t, vertices_tt, gammas_tt);
}

} // namespace xuan