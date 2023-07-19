#include "DER.h"

namespace xuan {

void locateVertexHessian(std::vector<Eigen::Triplet<double>> &hess_triplets, int row, int col, const Eigen::Vector3d &vec) {
    hess_triplets.emplace_back(row, col, vec(0));
    hess_triplets.emplace_back(row, col + 1, vec(1));
    hess_triplets.emplace_back(row, col + 2, vec(2));
    hess_triplets.emplace_back(col, row, vec(0));
    hess_triplets.emplace_back(col + 1, row, vec(1));
    hess_triplets.emplace_back(col + 2, row, vec(2));
}

void DER::boundaryDBC(Eigen::SparseMatrix<double> &hess) {
    // handle the boundary
    assert(hess.rows() == vertices_.size() * 3 + gamma_.size());
    for (auto i : DBC_vertices_) {
        if (i >= 0 && i < vertices_.size()) {
            hess.row(i) *= 0;
            hess.row(i + 1) *= 0;
            hess.row(i + 2) *= 0;
            hess.col(i) *= 0;
            hess.col(i + 1) *= 0;
            hess.col(i + 2) *= 0;
            hess.coeffRef(i, i) = 1;
            hess.coeffRef(i + 1, i + 1) = 1;
            hess.coeffRef(i + 2, i + 2) = 1;
        }
    }
    for (auto i : DBC_gamma_) {
        if (i >= 0 && i < gamma_.size()) {
            hess.row(vertices_.size() * 3 + i) *= 0;
            hess.col(vertices_.size() * 3 + i) *= 0;
            hess.coeffRef(vertices_.size() * 3 + i, vertices_.size() * 3 + i) = 1;
        }
    }
}

double DER::computeStretchEnergy() {
    double energy = 0.0;
    for (int i = 0; i < edges_.size(); ++i) {
        energy += ks_ * std::pow(edges_[i].norm() / undeformed_edges_[i].norm() - 1, 2) * undeformed_edges_[i].norm();
    }
    return energy / 2.0;
}

void DER::computeStretchGradient(Eigen::VectorXd &grad) {
    std::vector<Eigen::Vector3d> d_x(vertices_.size(), Eigen::Vector3d::Zero());

    for (int i = 0; i < edges_.size(); ++i) {
        double exterior = ks_ * (edges_[i].norm() / undeformed_edges_[i].norm() - 1);
        d_x[i] -= exterior * edges_[i].normalized();
        d_x[i + 1] += exterior * edges_[i].normalized();
    }

    // handle the boundary
    for (auto i : DBC_vertices_) {
        if (i >= 0 && i < vertices_.size())
            d_x[i] = Eigen::Vector3d::Zero();
    }

    grad.resize(vertices_.size() * 3 + gamma_.size());
    for (int i = 0; i < vertices_.size(); ++i) {
        grad.segment<3>(i * 3) = d_x[i];
    }
}

void DER::computeStretchHessian(Eigen::SparseMatrix<double> &hess) {
    std::vector<Eigen::Triplet<double>> hess_triplets;
    for (int i = 0; i < edges_.size(); ++i) {
        Eigen::Matrix3d local = Eigen::Matrix3d::Identity() / undeformed_edges_[i].norm() - Eigen::Matrix3d::Identity() / edges_[i].norm() + edges_[i] * edges_[i].transpose() / std::pow(edges_[i].norm(), 3);
        makePD(local);
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                hess_triplets.push_back(Eigen::Triplet<double>(i * 3 + j, i * 3 + k, ks_ * local(j, k)));
                hess_triplets.push_back(Eigen::Triplet<double>((i + 1) * 3 + j, (i + 1) * 3 + k, ks_ * local(j, k)));
                hess_triplets.push_back(Eigen::Triplet<double>(i * 3 + j, (i + 1) * 3 + k, -ks_ * local(j, k)));
                hess_triplets.push_back(Eigen::Triplet<double>((i + 1) * 3 + j, i * 3 + k, -ks_ * local(j, k)));
            }
        }
    }

    hess.resize(vertices_.size() * 3 + gamma_.size(), vertices_.size() * 3 + gamma_.size());
    hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());
}

double DER::computeBendEnergy() {
    double energy = 0.0;
    for (int i = 1; i < vertices_.size() - 1; ++i) {
        // compute the discrete curvature
        // note that this is not equal to the integrated one
        Eigen::Vector2d kappa = (Eigen::Vector2d(curvature_[i].dot(material_frame2_[i - 1]), -curvature_[i].dot(material_frame1_[i - 1])) + Eigen::Vector2d(curvature_[i].dot(material_frame2_[i]), -curvature_[i].dot(material_frame1_[i]))) / 2.0;
        energy += kb_ / voronoi_length_[i] * (kappa - undeformed_kappa_[i]).squaredNorm();
    }
    return energy / 2.0;
}

void DER::computeBendGradient(Eigen::VectorXd &grad) {
    std::vector<Eigen::Vector3d> d_x(vertices_.size(), Eigen::Vector3d::Zero());

    for (int i = 1; i < vertices_.size() - 1; ++i) {
        double kappa_1 = curvature_[i].dot(material_frame2_[i - 1] + material_frame2_[i]) / 2.0;
        double kappa_2 = -curvature_[i].dot(material_frame1_[i - 1] + material_frame1_[i]) / 2.0;

        Eigen::Vector3d tilde_t = (tangents_[i - 1] + tangents_[i]) / (1 + tangents_[i - 1].dot(tangents_[i]));

        Eigen::Vector3d d_kappa1_ek_minus_1 = (-kappa_1 * tilde_t + tangents_[i].cross((material_frame2_[i - 1] + material_frame2_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i - 1].norm();
        Eigen::Vector3d d_kappa1_ek = (-kappa_1 * tilde_t - tangents_[i - 1].cross((material_frame2_[i - 1] + material_frame2_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i].norm();
        double exterior = 2 * kb_ / voronoi_length_[i] * (kappa_1 - undeformed_kappa_[i](0));
        d_x[i - 1] -= exterior * d_kappa1_ek_minus_1;
        d_x[i] += exterior * d_kappa1_ek_minus_1;
        d_x[i + 1] += exterior * d_kappa1_ek;
        d_x[i] -= exterior * d_kappa1_ek;

        Eigen::Vector3d d_kappa2_ek_minus_1 = (-kappa_2 * tilde_t - tangents_[i].cross((material_frame1_[i - 1] + material_frame1_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i - 1].norm();
        Eigen::Vector3d d_kappa2_ek = (-kappa_2 * tilde_t + tangents_[i - 1].cross((material_frame1_[i - 1] + material_frame1_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i].norm();
        exterior = 2 * kb_ / voronoi_length_[i] * (kappa_2 - undeformed_kappa_[i](1));
        d_x[i - 1] -= exterior * d_kappa2_ek_minus_1;
        d_x[i] += exterior * d_kappa2_ek_minus_1;
        d_x[i + 1] += exterior * d_kappa2_ek;
        d_x[i] -= exterior * d_kappa2_ek;
    }

    // handle the boundary
    for (auto i : DBC_vertices_) {
        if (i >= 0 && i < vertices_.size())
            d_x[i] = Eigen::Vector3d::Zero();
    }

    grad.resize(vertices_.size() * 3 + gamma_.size());
    for (int i = 0; i < vertices_.size(); ++i) {
        grad.segment<3>(i * 3) = d_x[i];
    }
}

void DER::computeBendHessian(Eigen::SparseMatrix<double> &hess) {
    std::vector<Eigen::Triplet<double>> hess_triplets;

    for (int i = 1; i < edges_.size(); ++i) {
        double kappa_1 = curvature_[i].dot(material_frame2_[i - 1] + material_frame2_[i]) / 2.0;
        double kappa_2 = -curvature_[i].dot(material_frame1_[i - 1] + material_frame1_[i]) / 2.0;

        Eigen::Vector3d tilde_t = (tangents_[i - 1] + tangents_[i]) / (1 + tangents_[i - 1].dot(tangents_[i]));
        double chi = 1 + tangents_[i - 1].dot(tangents_[i]);
        Eigen::Matrix3d d_tilde_t_ek_minus_1 = ((Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i - 1], tangents_[i - 1])) - tensorProduct3d(tilde_t, (Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i - 1], tangents_[i - 1])) * tangents_[i])) / (chi * edges_[i - 1].norm());
        Eigen::Matrix3d d_tilde_t_ek = ((Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i], tangents_[i])) - tensorProduct3d(tilde_t, (Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i], tangents_[i])) * tangents_[i - 1])) / (chi * edges_[i].norm());
        Eigen::Vector3d d_chi_ek_minius_1 = (Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i - 1], tangents_[i - 1])) * tangents_[i] / edges_[i - 1].norm();
        Eigen::Vector3d d_chi_ek = (Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i], tangents_[i])) * tangents_[i - 1] / edges_[i].norm();
        Eigen::Matrix3d d_tk_ek = (Eigen::Matrix3d::Identity() - tensorProduct3d(tangents_[i], tangents_[i])) / edges_[i].norm();

        Eigen::Vector3d d_kappa1_ek_minus_1 = (-kappa_1 * tilde_t + tangents_[i].cross((material_frame2_[i - 1] + material_frame2_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i - 1].norm();
        Eigen::Vector3d d_kappa1_ek = (-kappa_1 * tilde_t - tangents_[i - 1].cross((material_frame2_[i - 1] + material_frame2_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i].norm();
        Eigen::Matrix3d d_kappa1_ek_minus_1_ek_minus_1 = symmetricMatrix(tensorProduct3d(tilde_t + tangents_[i], d_kappa1_ek_minus_1) + kappa_1 * d_tilde_t_ek_minus_1) + symmetricMatrix(tensorProduct3d(1.0 / chi * tangents_[i].cross((material_frame2_[i - 1] + material_frame2_[i]) / chi), d_chi_ek_minius_1));
        d_kappa1_ek_minus_1_ek_minus_1 /= -1.0 / edges_[i - 1].norm();
        Eigen::Matrix3d d_kappa1_ek_ek = symmetricMatrix(tensorProduct3d(tilde_t + tangents_[i - 1], d_kappa1_ek) + kappa_1 * d_tilde_t_ek) + symmetricMatrix(tensorProduct3d(1.0 / chi * tangents_[i - 1].cross((material_frame2_[i - 1] + material_frame2_[i]) / chi), d_chi_ek));
        d_kappa1_ek_ek /= -1.0 / edges_[i].norm();
        Eigen::Matrix3d d_kappa1_ek_minus_1_ek = (tensorProduct3d(tilde_t, d_kappa1_ek) + kappa_1 * d_tilde_t_ek) + (tensorProduct3d(1.0 / chi * tangents_[i].cross((material_frame2_[i - 1] + material_frame2_[i]) / chi), d_chi_ek) + tensorProduct3d(d_tk_ek.transpose() * ((material_frame2_[i - 1] + material_frame2_[i]) / chi), Eigen::Vector3d::Ones()));
        d_kappa1_ek_minus_1_ek /= -1.0 / edges_[i - 1].norm();

        Eigen::Vector3d d_kappa2_ek_minus_1 = (-kappa_2 * tilde_t - tangents_[i].cross((material_frame1_[i - 1] + material_frame1_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i - 1].norm();
        Eigen::Vector3d d_kappa2_ek = (-kappa_2 * tilde_t + tangents_[i - 1].cross((material_frame1_[i - 1] + material_frame1_[i]) / (1 + tangents_[i - 1].dot(tangents_[i])))) / edges_[i].norm();
        Eigen::Matrix3d d_kappa2_ek_minus_1_ek_minus_1 = symmetricMatrix(tensorProduct3d(tilde_t + tangents_[i], d_kappa2_ek_minus_1) + kappa_2 * d_tilde_t_ek_minus_1) + symmetricMatrix(tensorProduct3d(1.0 / chi * tangents_[i].cross((material_frame1_[i - 1] + material_frame1_[i]) / chi), d_chi_ek_minius_1));
        d_kappa2_ek_minus_1_ek_minus_1 /= -1.0 / edges_[i - 1].norm();
        Eigen::Matrix3d d_kappa2_ek_ek = symmetricMatrix(tensorProduct3d(tilde_t + tangents_[i - 1], d_kappa2_ek) + kappa_2 * d_tilde_t_ek) + symmetricMatrix(tensorProduct3d(1.0 / chi * tangents_[i - 1].cross((material_frame1_[i - 1] + material_frame1_[i]) / chi), d_chi_ek));
        d_kappa2_ek_ek /= -1.0 / edges_[i].norm();
        Eigen::Matrix3d d_kappa2_ek_minus_1_ek = (tensorProduct3d(tilde_t, d_kappa2_ek) + kappa_2 * d_tilde_t_ek) + (tensorProduct3d(1.0 / chi * tangents_[i].cross((material_frame1_[i - 1] + material_frame1_[i]) / chi), d_chi_ek) + tensorProduct3d(d_tk_ek.transpose() * ((material_frame1_[i - 1] + material_frame1_[i]) / chi), Eigen::Vector3d::Ones()));
        d_kappa2_ek_minus_1_ek /= -1.0 / edges_[i - 1].norm();

        Eigen::Matrix3d ek_minus_1_ek_minus_1 = d_kappa1_ek_minus_1 * d_kappa1_ek_minus_1.transpose() + (kappa_1 - undeformed_kappa_[i](0)) * d_kappa1_ek_minus_1_ek_minus_1 + d_kappa2_ek_minus_1 * d_kappa2_ek_minus_1.transpose() + (kappa_2 - undeformed_kappa_[i](1)) * d_kappa2_ek_minus_1_ek_minus_1;
        ek_minus_1_ek_minus_1 *= 2 * kb_ / voronoi_length_[i];
        makePD(ek_minus_1_ek_minus_1);
        Eigen::Matrix3d ek_ek = d_kappa1_ek * d_kappa1_ek.transpose() + (kappa_1 - undeformed_kappa_[i](0)) * d_kappa1_ek_ek + d_kappa2_ek * d_kappa2_ek.transpose() + (kappa_2 - undeformed_kappa_[i](1)) * d_kappa2_ek_ek;
        ek_ek *= 2 * kb_ / voronoi_length_[i];
        makePD(ek_ek);
        Eigen::Matrix3d ek_minus_1_ek = d_kappa1_ek_minus_1 * d_kappa1_ek.transpose() + (kappa_1 - undeformed_kappa_[i](0)) * d_kappa1_ek_minus_1_ek + d_kappa2_ek_minus_1 * d_kappa2_ek.transpose() + (kappa_2 - undeformed_kappa_[i](1)) * d_kappa2_ek_minus_1_ek;
        ek_minus_1_ek *= 2 * kb_ / voronoi_length_[i];
        makePD(ek_minus_1_ek);

        for (int j = 0; j < 3; ++j) {
            locateVertexHessian(hess_triplets, (i - 1) * 3, (i - 1) * 3, ek_minus_1_ek_minus_1.row(j));
            locateVertexHessian(hess_triplets, (i - 1) * 3, i * 3, -ek_minus_1_ek_minus_1.row(j));
            locateVertexHessian(hess_triplets, i * 3, i * 3, ek_minus_1_ek_minus_1.row(j));
            locateVertexHessian(hess_triplets, i * 3, i * 3, ek_ek.row(j));
            locateVertexHessian(hess_triplets, i * 3, (i + 1) * 3, -ek_ek.row(j));
            locateVertexHessian(hess_triplets, (i + 1) * 3, (i + 1) * 3, ek_ek.row(j));
            locateVertexHessian(hess_triplets, (i - 1) * 3, i * 3, ek_minus_1_ek.row(j));
            locateVertexHessian(hess_triplets, (i + 1) * 3, i * 3, ek_minus_1_ek.row(j));
            locateVertexHessian(hess_triplets, (i - 1) * 3, (i + 1) * 3, -ek_minus_1_ek.row(j));
            locateVertexHessian(hess_triplets, i * 3, i * 3, -ek_minus_1_ek.row(j));
        }
    }

    hess.resize(vertices_.size() * 3 + gamma_.size(), vertices_.size() * 3 + gamma_.size());
    hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());
}

double DER::computeTwistEnergy() {
    double energy = 0.0;
    for (int i = 1; i < edges_.size(); ++i) {
        double m_i = gamma_[i] - gamma_[i - 1] + reference_twist_[i];
        energy += kt_ * std::pow((m_i - undeformed_reference_twist_[i]), 2) / voronoi_length_[i];
    }
    return energy / 2.0;
}

void DER::computeTwistGradient(Eigen::VectorXd &grad) {
    std::vector<Eigen::Vector3d> d_x(vertices_.size(), Eigen::Vector3d::Zero());
    std::vector<double> d_gamma(gamma_.size(), 0.0);

    for (int i = 1; i < edges_.size(); ++i) {
        // compute the gradient w.r.t. gamma
        double m_i = gamma_[i] - gamma_[i - 1] + reference_twist_[i];
        double exterior = kt_ / voronoi_length_[i] * (m_i - undeformed_reference_twist_[i]);
        d_gamma[i] += exterior;
        d_gamma[i - 1] -= exterior;
        // compute the gradient w.r.t. x
        d_x[i - 1] += exterior * (-1.0) * curvature_[i] / edges_[i - 1].norm() / 2.0;
        d_x[i] += exterior * ((-1.0) * curvature_[i] / edges_[i].norm() / 2.0 + curvature_[i] / edges_[i - 1].norm() / 2.0);
        d_x[i + 1] += exterior * (curvature_[i] / edges_[i].norm() / 2.0);
    }

    // handle the boundary
    for (auto i : DBC_vertices_) {
        if (i >= 0 && i < vertices_.size())
            d_x[i] = Eigen::Vector3d::Zero();
    }
    for (auto i : DBC_gamma_) {
        if (i >= 0 && i < gamma_.size())
            d_gamma[i] = 0.0;
    }

    grad.resize(vertices_.size() * 3 + gamma_.size());
    for (int i = 0; i < vertices_.size(); ++i) {
        grad.segment<3>(i * 3) = d_x[i];
    }
    for (int i = 0; i < gamma_.size(); ++i) {
        grad(vertices_.size() * 3 + i) = d_gamma[i];
    }
}

// TODO: check validity
void DER::computeTwistHessian(Eigen::SparseMatrix<double> &hess) {
    std::vector<Eigen::Triplet<double>> hess_triplets;

    for (int i = 1; i < edges_.size(); ++i) {
        // compute the hessian w.r.t. gamma
        double exterior = kt_ / voronoi_length_[i];
        hess_triplets.emplace_back(vertices_.size() * 3 + i, vertices_.size() * 3 + i, exterior);
        hess_triplets.emplace_back(vertices_.size() * 3 + i - 1, vertices_.size() * 3 + i - 1, exterior);
        hess_triplets.emplace_back(vertices_.size() * 3 + i, vertices_.size() * 3 + i - 1, -exterior);
        hess_triplets.emplace_back(vertices_.size() * 3 + i - 1, vertices_.size() * 3 + i, -exterior);
        // compute the hessian w.r.t. gamma and x
        Eigen::Vector3d d_x_i_plus_1 = curvature_[i] / edges_[i].norm() / 2.0;
        Eigen::Vector3d d_x_i_minus_1 = -curvature_[i] / edges_[i - 1].norm() / 2.0;
        Eigen::Vector3d d_x_i = -d_x_i_plus_1 - d_x_i_minus_1;
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i, i * 3, d_x_i * exterior);
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i - 1, i * 3 + 1, -d_x_i * exterior);
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i, (i + 1) * 3, d_x_i_plus_1 * exterior);
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i - 1, (i + 1) * 3 + 1, -d_x_i_plus_1 * exterior);
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i, (i - 1) * 3, d_x_i_minus_1 * exterior);
        locateVertexHessian(hess_triplets, vertices_.size() * 3 + i - 1, (i - 1) * 3 + 1, -d_x_i_minus_1 * exterior);
        // compute the hessian w.r.t. x
        Eigen::Matrix3d hess_e_i_minus_1, hess_e_i, hess_e_i_e_i_minus_1;
        hess_e_i_minus_1 = symmetricMatrix(tensorProduct3d(curvature_[i], tangents_[i - 1]) +
                                           tensorProduct3d(curvature_[i], (tangents_[i] + tangents_[i - 1]) / (1 + tangents_[i].dot(tangents_[i - 1]))));
        hess_e_i_minus_1 *= -1 / (2 * edges_[i - 1].norm() * edges_[i - 1].norm());
        makePD(hess_e_i_minus_1);
        for (int j = 0; j < hess_e_i_minus_1.rows(); ++j) {
            locateVertexHessian(hess_triplets, (i - 1) * 3 + j, (i - 1) * 3, hess_e_i_minus_1.row(j) * exterior);
            locateVertexHessian(hess_triplets, (i - 1) * 3 + j, i * 3, -hess_e_i_minus_1.row(j) * exterior);
            locateVertexHessian(hess_triplets, i * 3 + j, i * 3, hess_e_i_minus_1.row(j) * exterior);
        }
        hess_e_i = symmetricMatrix(tensorProduct3d(curvature_[i], tangents_[i]) +
                                   tensorProduct3d(curvature_[i], (tangents_[i] + tangents_[i - 1]) / (1 + tangents_[i].dot(tangents_[i - 1]))));
        hess_e_i *= -1 / (2 * edges_[i].norm() * edges_[i].norm());
        makePD(hess_e_i);
        for (int j = 0; j < hess_e_i.rows(); ++j) {
            locateVertexHessian(hess_triplets, i * 3 + j, i * 3, hess_e_i.row(j) * exterior);
            locateVertexHessian(hess_triplets, i * 3 + j, (i + 1) * 3, -hess_e_i.row(j) * exterior);
            locateVertexHessian(hess_triplets, (i + 1) * 3 + j, (i + 1) * 3, hess_e_i.row(j) * exterior);
        }
        hess_e_i_e_i_minus_1 = skewSymmetricVector(tangents_[i - 1]) - tensorProduct3d(curvature_[i], (tangents_[i - 1] + tangents_[i]) / 2.0);
        hess_e_i_e_i_minus_1 *= -1 / (2 * edges_[i - 1].norm() * edges_[i].norm() * (1 + tangents_[i].dot(tangents_[i - 1])));
        makePD(hess_e_i_e_i_minus_1);
        for (int j = 0; j < hess_e_i_e_i_minus_1.rows(); ++j) {
            locateVertexHessian(hess_triplets, (i + 1) * 3 + j, i * 3, hess_e_i_e_i_minus_1.row(j) * exterior);
            locateVertexHessian(hess_triplets, (i - 1) * 3 + j, (i + 1) * 3, -hess_e_i_e_i_minus_1.row(j) * exterior);
            locateVertexHessian(hess_triplets, i * 3 + j, (i - 1) * 3, hess_e_i_e_i_minus_1.row(j) * exterior);
            locateVertexHessian(hess_triplets, i * 3 + j, i * 3, -hess_e_i_e_i_minus_1.row(j) * exterior);
        }
    }

    hess.resize(vertices_.size() * 3 + gamma_.size(), vertices_.size() * 3 + gamma_.size());
    hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());
}

double DER::computeEnergy(const Eigen::VectorXd &x_gamma) {
    double energy = 0.0;

    // elastic energy

    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> gamma;
    for (int i = 0; i < vertices_.size(); ++i) {
        vertices.push_back(vertices_[i] + Eigen::Vector3d(x_gamma[3 * i], x_gamma[3 * i + 1], x_gamma[3 * i + 2]));
    }
    for (int i = 0; i < gamma_.size(); ++i) {
        gamma.push_back(gamma_[i] + x_gamma[vertices_.size() / 3 + i]);
    }

    // update
    std::vector<Eigen::Vector3d> edges;
    edges.resize(vertices.size() - 1);
    std::vector<Eigen::Vector3d> tangents;
    tangents.resize(vertices.size() - 1);
    std::vector<Eigen::Vector3d> reference_frame1(reference_frame1_.size()), reference_frame2(reference_frame2_.size());
    std::vector<Eigen::Vector3d> material_frame1(material_frame1_.size()), material_frame2(material_frame1_.size());
    std::vector<Eigen::Vector3d> curvature(curvature_.size());
    std::vector<double> reference_twist(reference_twist_.size());
    for (int i = 0; i < vertices.size() - 1; ++i) {
        edges[i] = vertices[i + 1] - vertices[i];
        tangents[i] = edges[i].normalized();
        reference_frame1[i] = parallelTransport(reference_frame1_[i], tangents_[i], tangents[i]);
        reference_frame2[i] = parallelTransport(reference_frame2_[i], tangents_[i], tangents[i]);
        material_frame1[i] = rotate(reference_frame1[i], tangents[i], gamma[i]);
        material_frame2[i] = rotate(reference_frame2[i], tangents[i], gamma[i]);
        if (i > 0) {
            curvature[i] = 2.0 * tangents[i - 1].cross(tangents[i]) / (1 + tangents[i - 1].dot(tangents[i]));
            Eigen::Vector3d space_parallel_transport = parallelTransport(reference_frame1[i - 1], tangents[i - 1], tangents[i]);
            // signed angle from P_{t^{i-1}}^{t^i} a_1^{i-1} to a_1^i
            space_parallel_transport.normalize();
            double cos_angle = space_parallel_transport.dot(reference_frame1[i].normalized());
            double sin_angle = space_parallel_transport.cross(reference_frame1[i].normalized()).norm();
            reference_twist[i] = std::atan2(sin_angle, cos_angle);
        }
    }

    // stretch
    for (int i = 0; i < edges.size(); ++i) {
        energy += ks_ * std::pow(edges[i].norm() / undeformed_edges_[i].norm() - 1, 2) * undeformed_edges_[i].norm();
    }

    for (int i = 1; i < vertices_.size() - 1; ++i) {
        // bend
        Eigen::Vector2d kappa = (Eigen::Vector2d(curvature[i].dot(material_frame2[i - 1]), -curvature[i].dot(material_frame1[i - 1])) + Eigen::Vector2d(curvature[i].dot(material_frame2[i]), -curvature[i].dot(material_frame1[i]))) / 2.0;
        energy += kb_ / voronoi_length_[i] * (kappa - undeformed_kappa_[i]).squaredNorm();
        // twist
        double m_i = gamma[i] - gamma[i - 1] + reference_twist[i];
        energy += kt_ * std::pow((m_i - undeformed_reference_twist_[i]), 2) / voronoi_length_[i];
    }

    // kinetic energy
    for (int i = 0; i < vertices.size(); ++i) {
        Eigen::Vector3d x_hat = x_t_.segment<3>(3 * i) + h_ * velocity_[i] + h_ * h_ * mass_[i] * gravity_ / mass_[i];
        energy += mass_[i] * (vertices[i] - x_hat).squaredNorm();
    }

    return energy / 2.0;
}

bool DER::solve() {
    spdlog::info("Solving DER...");
    spdlog::info("Elastic energy before optimization: {}", computeElasticEnergy());
    int iter = 0;
    double p_norm_pre = 0.0;
    Eigen::VectorXd p;
    p.resize(vertices_.size() * 3 + gamma_.size());
    Eigen::VectorXd x;
    x.resize(vertices_.size() * 3 + gamma_.size());
    double alpha = 1.0;
    for (int i = 0; i < vertices_.size(); ++i) {
        x.segment<3>(3 * i) = vertices_[i];
    }
    for (int i = 0; i < gamma_.size(); ++i) {
        x[vertices_.size() * 3 + i] = gamma_[i];
    }

    while (1) {
        spdlog::info("Iteration: {}", iter++);
        Eigen::SparseMatrix<double> hess, hess_stretch, hess_bend, hess_twist;
        Eigen::VectorXd grad, grad_stretch, grad_bend, grad_twist;

        computeStretchGradient(grad_stretch);
        computeBendGradient(grad_bend);
        computeTwistGradient(grad_twist);
        grad = grad_stretch + grad_bend + grad_twist;
        for (int i = 0; i < vertices_.size(); ++i) {
            grad.segment<3>(3 * i) += mass_[i] * (vertices_[i] - x_t_.segment<3>(3 * i) + h_ * velocity_[i] + h_ * h_ * mass_[i] * gravity_ / mass_[i]);
        }

        computeStretchHessian(hess_stretch);
        computeBendHessian(hess_bend);
        computeTwistHessian(hess_twist);
        hess = hess_stretch + hess_bend + hess_twist;
        for (int i = 0; i < vertices_.size(); ++i) {
            hess.coeffRef(3 * i, 3 * i) += mass_[i];
            hess.coeffRef(3 * i, 3 * i + 1) += mass_[i];
            hess.coeffRef(3 * i, 3 * i + 2) += mass_[i];
            hess.coeffRef(3 * i + 1, 3 * i) += mass_[i];
            hess.coeffRef(3 * i + 1, 3 * i + 1) += mass_[i];
            hess.coeffRef(3 * i + 1, 3 * i + 2) += mass_[i];
            hess.coeffRef(3 * i + 2, 3 * i) += mass_[i];
            hess.coeffRef(3 * i + 2, 3 * i + 1) += mass_[i];
            hess.coeffRef(3 * i + 2, 3 * i + 2) += mass_[i];
        }
        boundaryDBC(hess);

        writeGradHessian(grad, hess, "grad_hess.txt");

        int max_iter = 1000;
        EigenConjugateGradient solver(max_iter);
        if (!solver.init(hess))
            return false;

        if (!solver.solve(-grad, p))
            return false;

        if (fabs(p.lpNorm<Eigen::Infinity>() - p_norm_pre) < 1e-6)
            break;
        p_norm_pre = p.lpNorm<Eigen::Infinity>();

        // line search
        double energy_pre = computeEnergy(x);
        double energy = computeEnergy(x + alpha * p);
        while (energy > energy_pre) {
            alpha /= 2.0;
            energy_pre = energy;
            energy = computeEnergy(x + alpha * p);
        }
        energy_pre = energy;
        x += alpha * p;
        update(alpha * p);
    }

    for (int i = 0; i < vertices_.size(); ++i) {
        x_t_.segment<3>(3 * i) = vertices_[i];
    }

    spdlog::info("Elastic energy after optimization: {}", computeElasticEnergy());

    return true;
}

} // namespace xuan
