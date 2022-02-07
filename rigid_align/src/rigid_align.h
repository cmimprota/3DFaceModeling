#ifndef __rigd_align__
#define __rigd_align__

#include <Eigen/Dense>

// Computes rotation between sets of 3d landmarks and returns it as a rotation martix
// Uses the Kabsch algorithm (with SVD)
void get_rigid_transform(const Eigen::MatrixXd& source_landmarks, const Eigen::MatrixXd& target_landmarks, Eigen::MatrixXd& rotation);

// Scales template verticies
void get_scaled_template(const Eigen::MatrixXd &V_template, const Eigen::MatrixXd& source_landmarks, const Eigen::MatrixXd& target_landmarks, Eigen::MatrixXd& V_scaled_template);

// Aligns face to template
void get_aligned_face(const Eigen::MatrixXd& V_source, const Eigen::MatrixXd& source_landmarks, const Eigen::MatrixXd& target_landmarks, Eigen::MatrixXd& V_aligned_source);

#endif
