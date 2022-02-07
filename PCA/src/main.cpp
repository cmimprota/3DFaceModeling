#include <iostream>
#include <fstream>
#include <vector>

#include <experimental/filesystem> // running c++ 11, so #include <filesystem> does not work yet

#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <Eigen/Eigenvalues>

#include <Spectra/SymEigsBase.h>
#include <Spectra/SymEigsSolver.h>

using namespace std;
using namespace Eigen;

using Viewer = igl::opengl::glfw::Viewer;
Viewer viewer;

string load_folder_name = "../warped_out_small/";
//string load_folder_name = "../warped_out_small_v2/";
string store_file_name = "../results/";

Eigen::MatrixXd V;
// Store vertices for ALL loaded meshes
MatrixXd visages; // each column is a mesh

// Store the mean visage
VectorXd mean_visage;

Eigen::MatrixXi F;
// Store faces for a sample loaded meshes
MatrixXi F_sample;

int number_of_meshes;
int number_of_features;

/* Eigen visages */
int number_of_eigen_visages = 8;

MatrixXd eigen_visages; // eigen vectors
VectorXd eigen_values;

VectorXd eigen_variance;

int slider_limit = 100;
float eigenface1 = 0;
float eigenface2 = 0;
float eigenface3 = 0;
float eigenface4 = 0;
float eigenface5 = 0;
float eigenface6 = 0;
float eigenface7 = 0;
float eigenface8 = 0;

/* Morphing */
MatrixXd morphing_weights;

int morphing_face_1 = 1;
int morphing_face_2 = 2;

float morphing_slider = 0.;

VectorXd weights_face_1;
VectorXd weights_face_2;
VectorXd morphing_parameters;

void computeEigenVisages(){
    
    // Compute covariance matrix
	MatrixXd deviation = visages.colwise() - mean_visage;
    MatrixXd temp = (deviation * deviation.transpose());
	MatrixXd covariance = temp / (number_of_meshes - 1);
    
    // https://spectralib.org/doc/classspectra_1_1symeigssolver
	Spectra::DenseSymMatProd<double> operation(covariance);
	Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double>> eigen_solver(&operation, number_of_eigen_visages, number_of_eigen_visages*2);
	
    // Perform Eigen decomposition
    eigen_solver.init(); 
	int _t = eigen_solver.compute();
   
	// Retrieve results
	eigen_visages.resize(number_of_features, number_of_eigen_visages);
	if (eigen_solver.info() == Spectra::SUCCESSFUL) {
		eigen_values = eigen_solver.eigenvalues();
		eigen_visages = eigen_solver.eigenvectors();
        cout << "Eigen decomposition: successful" << endl;
	} 
    else { cout << "Eigen decomposition: UNsuccessful" << endl; }

	// Compute the variance for each eigenvalue - for GUI
    float variance = covariance.diagonal().sum();
    
    eigen_variance.resize(number_of_eigen_visages);
    for(int i = 0; i < number_of_eigen_visages; i++){
		eigen_variance[i] = eigen_values[i] / variance * 100;
        //eigen_variance[i] *= 100; // %
	}

    // Compute eigen coordinates per sample - for Morphing
    morphing_weights = (deviation.transpose() * eigen_visages).transpose();
    
	cout << "Eigen visages computed" << endl;

}

MatrixXd matrixFromVector(VectorXd& V_vector){
    
    int V_r = V_vector.size() / 3;

	MatrixXd V_mat = MatrixXd::Zero(V_r, 3);

	V_mat.col(0) = V_vector.head(V_r);
	V_mat.col(1) = (V_vector.tail(2 * V_r)).head(V_r); // trick
	V_mat.col(2) = V_vector.tail(V_r);

	return V_mat;
}

void computeEigenParametrization(){
	VectorXd parameterization = mean_visage +
	                            ((eigenface1 * eigen_visages.col(0)) +
                                (eigenface2 * eigen_visages.col(1)) +
                                (eigenface3 * eigen_visages.col(2)) +
                                (eigenface4 * eigen_visages.col(3)) +
                                (eigenface5 * eigen_visages.col(4)) +
                                (eigenface6 * eigen_visages.col(5)) +
                                (eigenface7 * eigen_visages.col(6)) +
                                (eigenface8 * eigen_visages.col(7)));
	// Draw
    V = matrixFromVector(parameterization);
	F = F_sample;

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(false);
    //viewer.core.align_camera_center(V);
}

void computeMorphingParametrization(){

	weights_face_1 = morphing_weights.col(morphing_face_1 - 1);
	weights_face_2 = morphing_weights.col(morphing_face_2 - 1);

	morphing_parameters = weights_face_1 + morphing_slider * (weights_face_2 - weights_face_1);

	VectorXd parameterization = mean_visage +
                                ((morphing_parameters(0) * eigen_visages.col(0))+
                                (morphing_parameters(1) * eigen_visages.col(1))+
                                (morphing_parameters(2) * eigen_visages.col(2))+
                                (morphing_parameters(3) * eigen_visages.col(3))+
                                (morphing_parameters(4) * eigen_visages.col(4))+
                                (morphing_parameters(5) * eigen_visages.col(5))+
                                (morphing_parameters(6) * eigen_visages.col(6))+
                                (morphing_parameters(7) * eigen_visages.col(7)));

	// Draw
    V = matrixFromVector(parameterization);
	F = F_sample;

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(false);
    //viewer.core.align_camera_center(V);

}


void computeReconstructionError(){
	
	// Compute mean reconstruction error
	double sum_mean_errors = 0;
	for (size_t i = 0; i < visages.cols(); i++){
		// Reconstruct face
		VectorXd parameters = morphing_weights.col(i);
		VectorXd parameterization = mean_visage +
									((parameters(0) * eigen_visages.col(0))+
									(parameters(1) * eigen_visages.col(1))+
									(parameters(2) * eigen_visages.col(2))+
									(parameters(3) * eigen_visages.col(3))+
									(parameters(4) * eigen_visages.col(4))+
									(parameters(5) * eigen_visages.col(5))+
									(parameters(6) * eigen_visages.col(6))+
									(parameters(7) * eigen_visages.col(7)));
		
		// Compute L1 distance
		VectorXd diff = (parameterization - visages.col(i)).cwiseAbs();
		sum_mean_errors += diff.mean();
		//cout << "Current mean reconstruction error: "<< diff.mean() << endl;
	}
	// Average errors
	sum_mean_errors /= visages.cols();
	cout << "Mean reconstruction error: "<< sum_mean_errors << endl;
}


bool load_mesh(Viewer& viewer, string load_folder_name, Eigen::MatrixXd& V, Eigen::MatrixXi& F){
    
    // Load ALL meshes in folder
    std::vector<std::string> filenames;
    namespace std_filesystem = std::experimental::filesystem; ;// as I'm running c++ 11

    // Find all filenames in directory
    const std_filesystem::directory_iterator end{};
	for (std_filesystem::directory_iterator iterator{ load_folder_name }; iterator != end; ++iterator){
        // check if regular file AND has ".obj" extention
        // http://en.cppreference.com/w/cpp/experimental/fs/is_regular_file 
		if ((std_filesystem::is_regular_file(*iterator)) && (iterator->path().extension() == ".obj")){
            filenames.push_back(iterator->path().string());
        }
	}

    cout << filenames.size() << " filenames found in folder " << load_folder_name << endl;

    std::vector<VectorXd> V_vector;
    // Load meshes from all filenames
    for(int i = 0; i < filenames.size(); i++){
        igl::read_triangle_mesh(filenames[i], V, F);
        V_vector.push_back(Map<VectorXd>(V.data(), V.cols() * V.rows()));
        if(i == 0)
		    F_sample = F;
    }

    number_of_meshes = V_vector.size();
    number_of_features = V_vector.front().size();

    visages.resize(number_of_features, number_of_meshes);
	for (int i = 0; i < number_of_meshes; i++){
		visages.col(i) = V_vector[i]; // each mesh is a column
	}

    cout << "Visages are loaded" << endl;
    
    return true;
}

int main(int argc, char *argv[]) {
    
    /* LOAD MESHES */
    load_mesh(viewer,load_folder_name,V,F);

    /* MEAN VISAGE */
	mean_visage.resize(number_of_features);
	mean_visage = visages.rowwise().mean();
    
    /* EIGEN VISAGES*/
    computeEigenVisages();
	
	/* RECONSTRUCTION ERROR */
	computeReconstructionError();
	
	/* SHOW MEAN FACE */
	V = matrixFromVector(mean_visage);
	F = F_sample;
	viewer.data().clear();
	viewer.data().set_mesh(V, F);
	viewer.data().set_face_based(false);
	viewer.core.align_camera_center(V);
	
	
	
    /* MENU */
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	menu.callback_draw_viewer_menu = [&](){

        // Add new menu group
        if (ImGui::CollapsingHeader("PCA", ImGuiTreeNodeFlags_DefaultOpen)){

            if (ImGui::CollapsingHeader("Mean Visage", ImGuiTreeNodeFlags_DefaultOpen)){
                if (ImGui::Button("Show Mean Visage", ImVec2(-1, 0))){
			        V = matrixFromVector(mean_visage);
	                F = F_sample;

                    viewer.data().clear();
	                viewer.data().set_mesh(V, F);
	                viewer.data().set_face_based(false);
	                viewer.core.align_camera_center(V);
		        }
            }

            if (ImGui::CollapsingHeader("Eigenfaces", ImGuiTreeNodeFlags_DefaultOpen)){

                // Sliders for eigen faces weights
                if (ImGui::SliderFloat("Eigenface 1", &eigenface1,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF1 (%)", (float)(eigen_variance[0]), "%.2f");
                    
                if (ImGui::SliderFloat("Eigenface 2", &eigenface2,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF2 (%)", (float)(eigen_variance[1]), "%.2f");

                if (ImGui::SliderFloat("Eigenface 3", &eigenface3,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF3 (%)", (float)(eigen_variance[2]), "%.2f");
                
                if (ImGui::SliderFloat("Eigenface 4", &eigenface4,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF4 (%)", (float)(eigen_variance[3]), "%.2f");

                if (ImGui::SliderFloat("Eigenface 5", &eigenface5,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF5 (%)", (float)(eigen_variance[4]), "%.2f");
                
                if (ImGui::SliderFloat("Eigenface 6", &eigenface6,  -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF6 (%)", (float)(eigen_variance[5]), "%.2f");
                
                if (ImGui::SliderFloat("Eigenface 7", &eigenface7, -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF7 (%)", (float)(eigen_variance[6]), "%.2f");

                if (ImGui::SliderFloat("Eigenface 8", &eigenface8, -slider_limit, slider_limit, "%.2f"))
                    computeEigenParametrization();
                    ImGui::Value("Variance EF8 (%)", (float)(eigen_variance[7]), "%.2f");
                
                if (ImGui::Button("Reset Eigen Visage", ImVec2(-1, 0))){
                    
                    eigenface1 = 0.0;
                    eigenface2 = 0.0;
                    eigenface3 = 0.0;
                    eigenface4 = 0.0;
                    eigenface5 = 0.0;
                    eigenface6 = 0.0;
                    eigenface7 = 0.0;
                    eigenface8 = 0.0;

                    computeEigenParametrization(); // reset to mean visage
                }
            }

            if (ImGui::CollapsingHeader("Morphing", ImGuiTreeNodeFlags_DefaultOpen)){
                
                if (ImGui::InputInt("Visage 1", &morphing_face_1)){
                    
                    // needs to be within number of meshes
				    if (morphing_face_1 > number_of_meshes)
					    morphing_face_1 = number_of_meshes;
				    
				    if (morphing_face_1 < 1)
                        morphing_face_1 = 1;
				        
				    computeMorphingParametrization();
                }

                if (ImGui::InputInt("Visage 2", &morphing_face_2)){
                    // needs to be within number of meshes
				    if (morphing_face_2 > number_of_meshes)
					    morphing_face_2 = number_of_meshes;
				    
				    if (morphing_face_2 < 1)
                        morphing_face_2 = 1;
				        
				    computeMorphingParametrization();
                }

                if (ImGui::SliderFloat("Morphing", &morphing_slider, 0., 1., "%.2f"))
					computeMorphingParametrization();
				
			}

            if (ImGui::CollapsingHeader("Save", ImGuiTreeNodeFlags_DefaultOpen)){
                if (ImGui::Button("Save to OBJ", ImVec2(-1, 0))){
                    //store mesh in OBJ format
                    igl::writeOBJ(store_file_name + "PCA.obj", V, F);
                    cout << "Current object stored in " << store_file_name << "PCA.obj" << endl;
                }
            }
        }
	};

    viewer.launch();
}
