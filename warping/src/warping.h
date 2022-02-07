#ifndef ASSIGNMENT1_WARPING_H
#define ASSIGNMENT1_WARPING_H

#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>

void generateConstraints(std::map<int, int>& landmarks_template, std::map<int, int>& landmarks_target, Eigen::MatrixXd& V, Eigen::MatrixXd& V_target, Eigen::MatrixXi& F, Eigen::SparseMatrix<double> &C, Eigen::MatrixXd &d);

Eigen::MatrixXd computeWarping(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& V_target, std::map<int, int>& landmarks_template, std::map<int, int>& landmarks_target);

#endif //ASSIGNMENT1_WARPING_H
