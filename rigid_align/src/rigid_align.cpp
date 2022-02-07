#include "rigid_align.h"

#include <iostream>
#include <fstream>

#include <igl/slice.h>


using namespace igl;
using namespace std;
using namespace Eigen;


// Computes rotation between sets of 3d landmarks and returns it as a rotation martix
void get_rigid_transform(const MatrixXd& source_landmarks, const MatrixXd& target_landmarks, MatrixXd& rotation) {

	//MatrixXd H = source_landmarks.transpose() * target_landmarks;
	MatrixXd H = target_landmarks.transpose() * source_landmarks;
	
	JacobiSVD<MatrixXd> svd(H, ComputeThinU | ComputeThinV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();

	double d = (V * U.transpose()).determinant();

	MatrixXd S2 = MatrixXd::Identity(3, 3);
	S2(2, 2) = d;
	rotation = V * S2 * U.transpose();

	assert(rotation.cols() == 3);
	assert(rotation.rows() == 3);
}

// Scales template verticies
void get_scaled_template(const MatrixXd& V_target, const MatrixXd& source_landmarks, const MatrixXd& target_landmarks, MatrixXd& V_scaled_target) {

	// Compute average distance to landmark centroid for both landmark sets
	RowVectorXd source_LM_centroid = source_landmarks.colwise().mean();
	RowVectorXd source_LM_dists = source_landmarks - source_LM_centroid;
	double source_LM_avgdist = (source_LM_dists.rowwise().norm()).mean();

	RowVectorXd target_LM_centroid = target_landmarks.colwise().mean();
	RowVectorXd target_LM_dists = target_landmarks - target_LM_centroid;
	double target_LM_avgdist = (target_LM_dists.rowwise().norm()).mean();

	// Change verticies to have a mean of (0,0,0)
	// Also move landmarks by same amount to keep them accurate
	V_scaled_target = V_target;
	MatrixXd target_landmarks_cp = target_landmarks;
	RowVectorXd vertex_centroid = V_scaled_target.colwise().mean();
	V_scaled_target.rowwise() -= vertex_centroid;
	target_landmarks_cp.rowwise() -= vertex_centroid;

	
	// Scale mesh by factor to get same size as source
	V_scaled_target *=  source_LM_avgdist / target_LM_avgdist;
	target_landmarks_cp *= source_LM_avgdist / target_LM_avgdist;

	// Translate to have landmark centroid at (0,0,0)
	V_scaled_target.rowwise() -= target_landmarks_cp.colwise().mean();

}

// Aligns face to template
void get_aligned_face(const MatrixXd& V_source, const MatrixXd& source_landmarks, const MatrixXd& target_landmarks, MatrixXd& V_aligned_source) {


	V_aligned_source = V_source;
	MatrixXd source_landmarks_c = source_landmarks;
	MatrixXd target_landmarks_c = target_landmarks;

	// Translate to have landmark centroids at (0,0,0)
	RowVectorXd source_LM_centroid = source_landmarks.colwise().mean();
	source_landmarks_c.rowwise() -= source_LM_centroid;
	V_aligned_source.rowwise() -= source_LM_centroid;

	RowVectorXd target_LM_centroid = target_landmarks.colwise().mean();
	target_landmarks_c.rowwise() -= target_LM_centroid;

	// Landmark sets have to have mean zero to compute rotation
	MatrixXd rotation;
	get_rigid_transform(source_landmarks_c, target_landmarks_c, rotation);

	V_aligned_source *= rotation;
}

