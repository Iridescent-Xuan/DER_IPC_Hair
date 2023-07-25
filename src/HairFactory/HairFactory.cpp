#include "HairFactory.h"

#include <spdlog/spdlog.h>
#include <fstream>
#include <set>

namespace xuan {
void HairFactory::generateOneHairSrand(const Eigen::Vector3d &root, const Eigen::Vector3d &center_line, int num_vertices, double length, double radius, double curliness) {
    assert(num_vertices >= 2 && radius >= 0 && length > 0 && center_line.norm() > 0);

    std::vector<Eigen::Vector3d> vertices(num_vertices);
    if (radius < 1e-6) { // straight hair
        double step = length / (num_vertices - 1);
        for (int i = 0; i < num_vertices; ++i) {
            vertices[i] = root + i * step * center_line.normalized();
        }
    } else { // curly hair
        assert(curliness > 0);
        double round = curliness * M_PI;
        Eigen::Vector3d helix, offset;
        for (int i = 0; i < num_vertices; ++i) {
            // first generate a cylindrical helix from (0, 0, 0) to (0, 0, length)
            double theta = double(i) / num_vertices;
            helix = Eigen::Vector3d(radius * cos(round * theta), radius * sin(round * theta), theta * length);
            if (i == 0) {
                offset = helix;
            }
            // then rotate the helix to align with centerline
            Eigen::Vector3d axis = -center_line.cross(Eigen::Vector3d(0, 0, 1));
            double angle = acos(center_line.normalized().dot(Eigen::Vector3d(0, 0, 1)));
            Eigen::AngleAxisd rotation(angle, axis);
            vertices[i] = root + rotation * (helix - offset);
        }
    }
    assert(root == vertices[0]);

    std::vector<double> gammas(num_vertices - 1, 0.0);
    std::vector<size_t> DBC_vertices{0};

    hairs_.emplace_back(DER(vertices, gammas, DBC_vertices, std::vector<size_t>()));
}

void HairFactory::generateHair(double radius, double curliness, int vertices_per_strand, double length, int resolution_theta, int resolution_phi) {
    assert(resolution_theta >= 1 && resolution_phi >= 3);
    V_head_.resize(resolution_theta * (resolution_phi - 1) + 2, 3);
    F_head_.resize(resolution_theta * (resolution_phi - 1) * 2, 3);

    int vertex_id = 0;
    V_head_.row(vertex_id++) = Eigen::Vector3d(0, 0, 1);
    for (int j = 1; j < resolution_phi; ++j) {
        for (int i = 0; i < resolution_theta; ++i) {
            Eigen::Vector3d sphere = Eigen::Vector3d(sin(M_PI * j / resolution_phi) * cos(2 * M_PI * i / resolution_theta),
                                                     sin(M_PI * j / resolution_phi) * sin(2 * M_PI * i / resolution_theta),
                                                     cos(M_PI * j / resolution_phi));
            Eigen::Vector3d normal = sphere.normalized();
            V_head_.row(vertex_id++) = sphere;
            if (i < resolution_theta / 2 && j < resolution_phi / 2) {
                generateOneHairSrand(sphere, normal, vertices_per_strand, length, radius, curliness);
                hair_roots_.push_back(vertex_id - 1);
            }
        }
    }
    V_head_.row(vertex_id++) = Eigen::Vector3d(0, 0, -1);
    assert(vertex_id == V_head_.rows());

    // generate triangles
    int face_id = 0;
    for (int j = 1; j < resolution_phi; ++j) {
        int theta_start = (j - 1) * resolution_theta + 1;
        int theta_end = j * resolution_theta;
        for (int i = theta_start; i <= theta_end; ++i) {
            int v1 = i;
            int v2 = i == theta_end ? theta_start : i + 1;
            int v3 = j == 1 ? 0 : v1 - resolution_theta;
            F_head_.row(face_id++) = Eigen::Vector3i(v1, v2, v3);
            v3 = j == resolution_phi - 1 ? V_head_.rows() - 1 : v2 + resolution_theta;
            F_head_.row(face_id++) = Eigen::Vector3i(v1, v3, v2);
        }
    }
    assert(face_id == F_head_.rows());

    // generate edges
    std::set<std::pair<size_t, size_t>> edge_set;
    for (int i = 0; i < F_head_.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            size_t v1 = F_head_(i, j);
            size_t v2 = F_head_(i, (j + 1) % 3);
            if (v1 > v2) {
                std::swap(v1, v2);
            }
            edge_set.insert(std::make_pair(v1, v2));
        }
    }
    E_head_.resize(edge_set.size(), 2);
    size_t edge_id = 0;
    for (const auto &edge : edge_set) {
        E_head_.row(edge_id++) = Eigen::Vector2i(edge.first, edge.second);
    }
    assert(edge_id == E_head_.rows());

    dhat_ = 0.0;
    num_variables_ = 0;
    num_hair_vertices_ = 0;
    num_hair_edges_ = 0;
    for (const auto &hair : hairs_) {
        dhat_ += hair.getRadius();
        num_variables_ += hair.numVariables();
        num_hair_vertices_ += hair.numVertices();
        num_hair_edges_ += hair.numEdges();
    }
    dhat_ /= hairs_.size();
    dhat_ *= num_hair_vertices_ / hairs_.size();

    generateCollisionMesh();

    spdlog::info("HairFactory::generateHair: {} hairs generated", hairs_.size());
}

void HairFactory::writeOBJ(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        spdlog::warn("HairFactory::writeOBJ: cannot open file {}", filename);
        return;
    }

    size_t offset = 0;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        out << "o hair" << i << "\n";
        hairs_[i].writeOBJ(out, offset);
        offset += hairs_[i].numVertices();
    }

    out << "o head\n";
    for (size_t i = 0; i < V_head_.rows(); ++i) {
        out << "v " << V_head_(i, 0) << " " << V_head_(i, 1) << " " << V_head_(i, 2) << "\n";
    }
    for (size_t i = 0; i < F_head_.rows(); ++i) {
        out << "f " << F_head_(i, 0) + 1 + offset << " " << F_head_(i, 1) + 1 + offset << " " << F_head_(i, 2) + 1 + offset << "\n";
    }

    out << std::endl;

    spdlog::debug("HairFactory::writeOBJ: {} hairs and 1 head written to {}", hairs_.size(), filename);

    out.close();
}

void HairFactory::deepUpdateOneHairStrand(const Eigen::VectorXd &x, size_t index_hair, double h) {
    assert(index_hair < hairs_.size());
    assert(x.size() == hairs_[index_hair].numVariables());
    hairs_[index_hair].deepUpdate(x, h);
}

void HairFactory::generateCollisionMesh() {

    Eigen::MatrixXd V; // head vertices + hair vertices - hair roots
    Eigen::MatrixXi F; // head triangles
    Eigen::MatrixXi E; // head edges + hair edges

    // Note that the hair root should not be included!
    V.resize(V_head_.rows() + num_hair_vertices_ - hairs_.size(), 3);
    // E.resize(E_head_.rows() + num_hair_edges_ * 3, 2);
    E.resize(E_head_.rows() + num_hair_edges_, 2);
    // F.resize(F_head_.rows() + num_hair_edges_, 3);
    F.resize(F_head_.rows(), 3);

    // fill the head
    V.topRows(V_head_.rows()) = V_head_;
    E.topRows(E_head_.rows()) = E_head_;
    F.topRows(F_head_.rows()) = F_head_;

    // fill the hairs
    size_t vertice_offset = V_head_.rows();
    size_t edge_offset = E_head_.rows();
    size_t face_offset = F_head_.rows();
    for (size_t i = 0; i < hairs_.size(); ++i) {
        DER hair = hairs_[i];
        std::vector<Eigen::Vector3d> vertices;
        hair.getVertices(vertices);
        E.row(edge_offset++) = Eigen::Vector2i(hair_roots_[i], vertice_offset);
        // E.row(edge_offset++) = Eigen::Vector2i(hair_roots_[i], hair_roots_[i]);
        // E.row(edge_offset++) = Eigen::Vector2i(vertice_offset, vertice_offset);
        // F.row(face_offset++) = Eigen::Vector3i(hair_roots_[i], vertice_offset, vertice_offset);
        for (size_t j = 1; j < vertices.size() - 1; ++j) {
            V.row(vertice_offset++) = vertices[j];
            E.row(edge_offset++) = Eigen::Vector2i(vertice_offset - 1, vertice_offset);
            // E.row(edge_offset++) = Eigen::Vector2i(vertice_offset - 1, vertice_offset - 1);
            // E.row(edge_offset++) = Eigen::Vector2i(vertice_offset, vertice_offset);
            // F.row(face_offset++) = Eigen::Vector3i(vertice_offset - 1, vertice_offset, vertice_offset);
        }
        V.row(vertice_offset++) = vertices.back();
    }
    assert(vertice_offset == V.rows());
    assert(edge_offset == E.rows());
    assert(face_offset == F.rows());

    collision_mesh_ = ipc::CollisionMesh(V, E, F);
    collision_constraints_.build(collision_mesh_, V, dhat_);

    if (collision_constraints_.compute_potential(collision_mesh_, V, dhat_) > 1e-6) {
        spdlog::warn("HairFactory::generateCollisionMesh: the initial state should be collision free!");
    }
}

Eigen::VectorXd HairFactory::moveHead(const Eigen::MatrixXd &head) {
    assert(head.rows() == V_head_.rows());

    Eigen::VectorXd xt;
    xt.resize(num_variables_);

    // fill the hairs
    size_t x_offset = 0;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        DER hair = hairs_[i];
        std::vector<Eigen::Vector3d> vertices;
        hair.getVertices(vertices);
        vertices[0] = head.row(hair_roots_[i]);
        for (size_t j = 0; j < vertices.size(); ++j) {
            xt.segment(x_offset, 3) = vertices[j];
            x_offset += 3;
        }
        std::vector<double> gammas;
        hair.getGammas(gammas);
        for (size_t j = 0; j < gammas.size(); ++j) {
            xt[x_offset++] = gammas[j];
        }
    }
    assert(x_offset == num_variables_);

    return xt;
}

Eigen::MatrixXd HairFactory::generateCollisionVertices(const Eigen::VectorXd &x, const Eigen::MatrixXd &head) {
    assert(x.size() == num_variables_ && head.rows() == V_head_.rows());

    Eigen::MatrixXd V;
    V.resize(V_head_.rows() + num_hair_vertices_ - hairs_.size(), 3);
    V.topRows(V_head_.rows()) = head;

    // fill the hairs from x
    size_t vertice_offset = V_head_.rows();
    size_t x_offset = 0;
    for (auto hair : hairs_) {
        // never add in the root
        x_offset += 3;
        for (size_t j = 1; j < hair.numVertices(); ++j) {
            V.row(vertice_offset++) = x.segment(x_offset, 3);
            x_offset += 3;
        }
        x_offset += hair.numEdges();
    }
    assert(vertice_offset == V.rows());
    assert(x_offset == x.size());

    return V;
}

double HairFactory::collisionEnergy(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints) {
    Eigen::MatrixXd V = generateCollisionVertices(x, head);
    if (update_constraints)
        collision_constraints_.build(collision_mesh_, V, dhat_);
    return kappa_ * collision_constraints_.compute_potential(collision_mesh_, V, dhat_);
}

std::vector<Eigen::VectorXd> HairFactory::collisionGradient(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints) {
    assert(x.size() == num_variables_ && head.rows() == V_head_.rows());

    Eigen::MatrixXd V = generateCollisionVertices(x, head);
    if (update_constraints)
        collision_constraints_.build(collision_mesh_, V, dhat_);

    Eigen::VectorXd grad = kappa_ * collision_constraints_.compute_potential_gradient(collision_mesh_, V, dhat_);

    std::vector<Eigen::VectorXd> grad_hairs;
    grad_hairs.resize(hairs_.size());
    size_t grad_offset = V_head_.rows() * 3;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        Eigen::VectorXd grad_hair;
        grad_hair.resize(hairs_[i].numVariables());
        grad_hair.setZero();
        grad_hair.segment(0, 3) = Eigen::Vector3d(0, 0, 0); // the root is fixed
        for (size_t j = 1; j < hairs_[i].numVertices(); ++j) {
            grad_hair.segment(3 * j, 3) = grad.segment(grad_offset, 3);
            grad_offset += 3;
        }
        grad_hairs[i] = grad_hair;
    }
    assert(grad_offset == grad.size());

    return grad_hairs;
}

std::vector<Eigen::SparseMatrix<double>> HairFactory::collisionHessian(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, bool update_constraints) {
    assert(x.size() == num_variables_ && head.rows() == V_head_.rows());

    Eigen::MatrixXd V = generateCollisionVertices(x, head);
    if (update_constraints)
        collision_constraints_.build(collision_mesh_, V, dhat_);

    std::vector<Eigen::SparseMatrix<double>> hessians;
    hessians.resize(hairs_.size());

    Eigen::SparseMatrix<double> hess = collision_constraints_.compute_potential_hessian(collision_mesh_, V, dhat_, true);

    size_t vertice_offset = V_head_.rows() * 3;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        Eigen::SparseMatrix<double> hess_hair;
        hess_hair.resize(hairs_[i].numVariables(), hairs_[i].numVariables());
        hess_hair.setZero();
        // TODO: this is quite slow, can be optimized
        for (size_t r = 1; r < hairs_[i].numVertices(); ++r) {
            for (size_t c = 1; c < hairs_[i].numVertices(); ++c) {
                for (int x = 0; x < 3; ++x) {
                    for (int y = 0; y < 3; ++y) {
                        hess_hair.coeffRef(3 * r + x, 3 * c + y) = hess.coeffRef(vertice_offset + 3 * (r - 1) + x, vertice_offset + 3 * (c - 1) + y);
                    }
                }
            }
        }
        vertice_offset += (hairs_[i].numVertices() - 1) * 3;
        hessians[i] = hess_hair;
    }
    assert(vertice_offset == hess.rows());

    return hessians;
}

double HairFactory::computeCollisionFreeStepSize(const Eigen::VectorXd &xt, const Eigen::VectorXd &xtt, const Eigen::MatrixXd &head) {
    assert(xt.size() == num_variables_ && xtt.size() == num_variables_ && head.rows() == V_head_.rows());

    Eigen::MatrixXd Vt = generateCollisionVertices(xt, head);
    Eigen::MatrixXd Vtt = generateCollisionVertices(xtt, head);

    return ipc::compute_collision_free_stepsize(collision_mesh_, Vt, Vtt);
}

void HairFactory::accumulateDerivatives(const Eigen::VectorXd &x, const Eigen::MatrixXd &head, double h,
                                        double &energy, std::vector<Eigen::VectorXd> &grad, std::vector<Eigen::SparseMatrix<double>> &hess) {
    assert(x.size() == num_variables_ && head.rows() == V_head_.rows());
    energy = 0.0;
    grad.clear();
    hess.clear();
    for (size_t i = 0; i < hairs_.size(); ++i) {
        grad.push_back(Eigen::VectorXd::Zero(hairs_[i].numVariables()));
        hess.push_back(Eigen::SparseMatrix<double>(hairs_[i].numVariables(), hairs_[i].numVariables()));
    }

    // collision
    spdlog::debug("Handling IPC collision...");
    spdlog::debug("----Building collision constraints...");
    Eigen::MatrixXd V = generateCollisionVertices(x, head);
    collision_constraints_.build(collision_mesh_, V, dhat_);
    spdlog::debug("----Computing collision derivatives...");
    double collision_energy = collisionEnergy(x, head, true);
    std::vector<Eigen::VectorXd> collision_grad = collisionGradient(x, head, false);
    std::vector<Eigen::SparseMatrix<double>> collision_hess = collisionHessian(x, head, false);

    energy += collision_energy;
    grad = collision_grad;
    hess = collision_hess;

    // DER
    spdlog::debug("Handling DER...");
    double hair_energy = 0.0;
    size_t hair_offset = 0;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        double hair_energy_i;
        Eigen::VectorXd hair_grad_i;
        Eigen::SparseMatrix<double> hair_hess_i;
        Eigen::VectorXd x_i;
        x_i.resize(hairs_[i].numVariables());
        x_i.segment(0, 3) = head.row(hair_roots_[i]);
        hair_offset += 3;
        for (size_t j = 1; j < hairs_[i].numVertices(); ++j) {
            x_i.segment(3 * j, 3) = x.segment(hair_offset, 3);
            hair_offset += 3;
        }
        for (size_t j = 0; j < hairs_[i].numEdges(); ++j) {
            x_i[hairs_[i].numVertices() * 3 + j] = x[hair_offset++];
        }

        hairs_[i].incrementalPotential(x_i, h, hair_energy_i, hair_grad_i, hair_hess_i);

        hair_energy += hair_energy_i;
        grad[i] += hair_grad_i;
        hess[i] += hair_hess_i;
    }
    assert(hair_offset == x.size());

    energy += hair_energy;
}

void HairFactory::deepUpdate(const Eigen::VectorXd &x, double h, const Eigen::MatrixXd &head) {
    assert(x.size() == num_variables_ && head.rows() == V_head_.rows());

    // update the head
    V_head_ = head;

    // update the hairs
    size_t x_offset = 0;
    for (size_t i = 0; i < hairs_.size(); ++i) {
        Eigen::VectorXd x_i;
        x_i.resize(hairs_[i].numVariables());
        x_i = x.segment(x_offset, hairs_[i].numVariables());
        hairs_[i].deepUpdate(x_i, h);
        x_offset += hairs_[i].numVariables();
    }
    assert(x_offset == x.size());

    generateCollisionMesh();
}

bool HairFactory::violateDBC(const Eigen::VectorXd &xt, const Eigen::VectorXd &xtt) {
    assert(xt.size() == num_variables_ && xtt.size() == num_variables_);

    std::vector<size_t> offset = hairOffset();
    for (size_t i = 0; i < hairs_.size(); ++i) {
        Eigen::VectorXd xt_i = xt.segment(offset[i], hairs_[i].numVariables());
        Eigen::VectorXd xtt_i = xtt.segment(offset[i], hairs_[i].numVariables());
        if (hairs_[i].violateDBC(xt_i, xtt_i)) {
            return true;
        }
    }
    return false;
}

Eigen::Vector3d HairFactory::hairBoundingBox() const {
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    double max_z = std::numeric_limits<double>::min();
    for (auto hair : hairs_) {
        Eigen::Vector3d bbox = hair.bboxSize();
        min_x = std::min(min_x, bbox[0]);
        min_y = std::min(min_y, bbox[1]);
        min_z = std::min(min_z, bbox[2]);
        max_x = std::max(max_x, bbox[0]);
        max_y = std::max(max_y, bbox[1]);
        max_z = std::max(max_z, bbox[2]);
    }
    return Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);
}

} // namespace xuan