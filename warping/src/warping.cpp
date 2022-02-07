#include "warping.h"

#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/boundary_loop.h>

using namespace igl;
using namespace std;
using namespace Eigen;

void generateConstraints(map<int, int>& landmarks_template, map<int, int>& landmarks_target, Eigen::MatrixXd& V, Eigen::MatrixXd& V_target, Eigen::MatrixXi& F, Eigen::SparseMatrix<double> &C, MatrixXd &d){
    std::vector<int> loop;
    igl::boundary_loop(F, loop);
    C.resize(landmarks_template.size()+loop.size(), V.rows());
    d.resize(landmarks_template.size()+loop.size(), 3);
    int i = 0;
    for (auto const& x : landmarks_template){
        int template_idx = x.first;
        int template_vertex = x.second;
        int target_vertex = landmarks_target[template_idx];
        C.insert(i, template_vertex) = 1;
        d.row(i) = V_target.row(target_vertex);
        i++;
    }
    // add boundary
    for (int vertex: loop){
        C.insert(i, vertex) = 1;
        d.row(i) = V.row(vertex);
        i++;
    }
}

Eigen::MatrixXd computeWarping(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& V_target, map<int, int>& landmarks_template, map<int, int>& landmarks_target){
    SparseMatrix<double> A;
    MatrixXd b;
    Eigen::SparseMatrix<double> C;
    MatrixXd d;

    generateConstraints(landmarks_template, landmarks_target, V, V_target, F, C, d);

    SparseMatrix<double> L, M, Minv;
    igl::cotmatrix(V, F,L);
//        igl::massmatrix(V_start,F_start,igl::MASSMATRIX_TYPE_VORONOI,M);
//        igl::invert_diag(M,Minv);
//        L = Minv * L;

    A = L.transpose() * L;

    MatrixXd b_first = L * V;
    b = L.transpose() * b_first;


    SparseMatrix<double> left, upper, lower, zeros(C.rows(), C.rows()), CT;
    CT = C.transpose();
    upper = igl::cat(2, A, CT);
    lower = igl::cat(2, C, zeros);
    left = igl::cat(1, upper, lower);
    MatrixXd x(left.rows(), 3);
    MatrixXd right(left.rows(), 3);
    right = igl::cat(1, b, d);

    SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > solver;
    left.makeCompressed();
    solver.analyzePattern(left);
    solver.factorize(left);
    x = solver.solve(right);
    cout << x.rows() << " " << x.cols() << endl;
    Eigen::MatrixXd output = x.topLeftCorner(V.rows(), 3);
    return output;
}