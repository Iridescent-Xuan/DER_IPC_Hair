#pragma once

#include <Eigen/Dense>

namespace xuan {

template <typename T, int Size>
Eigen::Matrix<T, Size, Size> outerProduct(const Eigen::Vector<T, Size> &v1, const Eigen::Vector<T, Size> &v2) {
    return v1 * v2.transpose();
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

} // namespace xuan