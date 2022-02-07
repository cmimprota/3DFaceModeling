
#include <iostream>
#include <fstream>
#include <vector>

#include <experimental/filesystem> // running c++ 11, so #include <filesystem> does not work yet
// don't forget to add link_libraries(stdc++fs) to cmake
//#include <filesystem>

#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice.h>

#include "rigid_align.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;
namespace fs = std::experimental::filesystem;

Viewer viewer;

// vertex array shown, #V x3
Eigen::MatrixXd V;
// face array shown, #F x3
Eigen::MatrixXi F;

// vertex array source, #V x3
Eigen::MatrixXd V_s;
Eigen::MatrixXd V_s_out;
// face array source, #F x3
Eigen::MatrixXi F_s;
// landmark array source, #V x3
Eigen::VectorXi L_s;
Eigen::MatrixXd L_s_pos;

// vertex array target, #V x3
Eigen::MatrixXd V_t;
Eigen::MatrixXd V_t_out;
// face array target, #F x3
Eigen::MatrixXi F_t;
// landmark array target, #V x3
Eigen::VectorXi L_t;
Eigen::MatrixXd L_t_pos;



void compute_folder(string folder_in, string folder_out) {

	string template_path = "../data/template/headtemplate_noneck.obj";
	string template_LM_path = "../data/template/headtemplate_noneck.txt";
	int nb_LM = 22;
	string template_output_suffix = "_template.obj";

	// Get all files in folder_in and generate output filenames
	vector<string> filenames_obj;
	vector<string> filenames_landmarks;
	vector<string> filenames_obj_out;
	vector<string> filenames_template_out;
	//vector<string> filenames_test;
	const fs::directory_iterator end{};
	for (fs::directory_iterator iterator{ folder_in }; iterator != end; ++iterator) {
		// check if regular file AND has ".obj" extention
		if ((fs::is_regular_file(*iterator)) && (iterator->path().extension() == ".obj")) {
			filenames_obj.push_back(iterator->path().string());
			filenames_obj_out.push_back(folder_out + iterator->path().filename().string());
			filenames_template_out.push_back(folder_out + iterator->path().stem().string() + template_output_suffix);
			//filenames_test.push_back(folder_out + "combined/"+ iterator->path().stem().string() + "_test.obj");
		}
		if ((fs::is_regular_file(*iterator)) && (iterator->path().extension() == ".txt")) {
			filenames_landmarks.push_back(iterator->path().string());
		}
	}
	cout << filenames_obj.size() << " .obj mesh files found in folder " << folder_in << endl;
	cout << filenames_landmarks.size() << " .txt landmark files found in folder " << folder_in << endl;

	if (filenames_obj.size() != filenames_landmarks.size() || filenames_obj.size() == 0) {
		cout << "Not valid file folder..." << endl;
		return;
	}

	// Load template files
	Eigen::MatrixXd V_t_temp;
	igl::readOBJ(template_path, V_t_temp, F_t);
	V_t = V_t_temp.leftCols(3); // Trim off normals
	cout << "Cols " << V_t.cols() << endl;
	L_t.resize(nb_LM);
	string line;
	int a, b;
	ifstream infile(template_LM_path.c_str());
	while (std::getline(infile, line)) {
		istringstream iss(line);
		if (!(iss >> a >> b)) { break; } // error
		//b--; // index from 0
		L_t(b) = a;
	}
	igl::slice(V_t, L_t, 1, L_t_pos);

	// Process all files
	cout << "Starting processing of files..." << endl;
	for (size_t i = 0; i < filenames_obj.size(); i++){
		cout << "Starting set "<< i+1<<"/"<<filenames_obj.size() << endl;

		cout << "Reading " << filenames_obj.at(i) << endl;
		igl::readOBJ(filenames_obj.at(i), V_s, F_s);
		assert(V_s.cols() == 3); // Not the case for example (includes 3 more columns of normals)
		
		cout << "Reading " << filenames_landmarks.at(i) << endl;
		L_s.resize(nb_LM);
		ifstream infile2(filenames_landmarks.at(i).c_str());
		while (std::getline(infile2, line)) {
			std::istringstream iss(line);
			if (!(iss >> a >> b)) { break; } // error
			//b--; // index from 0
			L_s(b) = a;
		}
		igl::slice(V_s, L_s, 1, L_s_pos);

		cout << "Processing..."<< endl;
		// Get scaled template vertex positions
		get_scaled_template(V_t, L_s_pos, L_t_pos, V_t_out);
		// Get aligned face vertex positions
		get_aligned_face(V_s, L_s_pos, L_t_pos, V_s_out);

		// Save files to folder_out
		cout << "Writing to " << filenames_obj_out.at(i) << endl;
		igl::writeOBJ(filenames_obj_out.at(i), V_s_out, F_s);
		
		cout << "Writing to " << filenames_template_out.at(i) << endl;
		igl::writeOBJ(filenames_template_out.at(i), V_t_out, F_t);


		// Just for testing
		// Join meshes into V F
		/*V.resize(V_t_out.rows() + V_s_out.rows(), 3);
		V << V_t_out, V_s_out;
		F.resize(F_t.rows() + F_s.rows(), 3);
		F << F_t, (F_s.array() + V_t_out.rows());

		cout << "Writing to " << filenames_test.at(i) << endl;
		igl::writeOBJ(filenames_test.at(i), V, F);*/
	}

}


void show_mesh(MatrixXd& mV, MatrixXi& mF) {
	viewer.data().clear();
	viewer.data().set_mesh(mV, mF);
	viewer.data().show_lines = false;
}

bool callback_key_pressed(Viewer& viewer, unsigned char key, int modifiers) {
	MatrixXd C;
	switch (key) {
	case '1':
		show_mesh(V_t, F_t);
		break;
	case '2':
		show_mesh(V_s, F_s);
		break;
	case '3':
		show_mesh(V_t_out, F_t); // Output template
		C.resize(F_t.rows(), 3);
		C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F_t.rows(), 1);
		viewer.data().set_colors(C);
		break;
	case '4':
		show_mesh(V_s_out, F_s); // Output head
		C.resize(F_s.rows(), 3);
		C << Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F_s.rows(), 1);
		viewer.data().set_colors(C);
		break;
	case '5':
		show_mesh(V, F); // Output template + head
		C.resize(F.rows(), 3);
		C <<Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F_t.rows(), 1),
			Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F_s.rows(), 1);
		viewer.data().set_colors(C);
		break;
	case ' ': // space bar
		break;
	}
	return true;
}

void compute_example() {

	// Source is a head mesh
	// Target is template

	// Read in meshes
	igl::readOBJ("../data/landmarks_example/headtemplate.obj", V_t, F_t);
	//igl::readOBJ("../data/landmarks_example/headtemplate_large.obj", V_t, F_t);

	Eigen::MatrixXd V_s_temp;
	igl::readOBJ("../data/landmarks_example/person0_.obj", V_s_temp, F_s);
	V_s = V_s_temp.leftCols(3);

	// Read in landmarks (adapted from warping branch)
	L_t.resize(23);
	L_s.resize(23);
	std::string line;
	int a, b;

	string landmark_str = "../data/landmarks_example/headtemplate_23landmarks";
	ifstream infile(landmark_str.c_str());
	while (std::getline(infile, line)){
		std::istringstream iss(line);
		if (!(iss >> a >> b)) { break; } // error
		b--; // index from 1
		L_t(b) = a;
	}

	landmark_str = "../data/landmarks_example/person0__23landmarks";
	ifstream infile2(landmark_str.c_str());
	while (std::getline(infile2, line)){
		std::istringstream iss(line);
		if (!(iss >> a >> b)) { break; } // error
		b--; // index from 1
		L_s(b) = a;
	}

	// Extract landmark positions
	igl::slice(V_t, L_t, 1, L_t_pos);
	igl::slice(V_s, L_s, 1, L_s_pos);

	// Get scaled template vertex positions
	get_scaled_template(V_t, L_s_pos, L_t_pos, V_t_out);
	// Get aligned face vertex positions
	get_aligned_face(V_s, L_s_pos, L_t_pos, V_s_out);

	//cout << "L_s:" << L_s << endl;
	//cout << "L_t:" << L_t << endl;
	//cout << "L_s_pos:" << L_s_pos << endl;
	//cout << "V_s:" << V_s.topRows(15) << endl;
	//cout << "V_s_out:" << V_s_out.topRows(15) << endl;
	//cout << "V_t_out:" << V_t_out.topRows(15) << endl;


	// Extract landmark positions at end (to make sure mean is 0)
	MatrixXd L_s_pos_out, L_t_pos_out;
	igl::slice(V_s_out, L_s, 1, L_s_pos_out);
	igl::slice(V_t_out, L_t, 1, L_t_pos_out);
	cout << "L_s_pos_out mean:" << L_s_pos_out.colwise().mean() << endl;
	cout << "L_t_pos_out mean:" << L_t_pos_out.colwise().mean() << endl;

	// Join meshes into V F
	V.resize(V_t_out.rows() + V_s_out.rows(), 3);
	V << V_t_out, V_s_out;
	F.resize(F_t.rows() + F_s.rows(), 3);
	F << F_t, (F_s.array() + V_t_out.rows());

	// Show
	show_mesh(V, F);
	viewer.core.align_camera_center(V, F);

	MatrixXd C(F.rows(), 3);
	C << Eigen::RowVector3d(0.2, 0.3, 0.8).replicate(F_t.rows(), 1),
		Eigen::RowVector3d(1.0, 0.7, 0.2).replicate(F_s.rows(), 1);
	viewer.data().set_colors(C);
	
	// Init viewer
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);
	viewer.callback_key_pressed = callback_key_pressed;
	viewer.launch();

}

int main(int argc,char *argv[]) {
	
	if (argc == 1) {
		cout << "No arguments, so showing example mesh" << endl;
		compute_example();
	}
	else if (argc == 3) {
		cout << "Two arguments, so computing over given folder" << endl;
		compute_folder(argv[1], argv[2]);
	}
	else {
		cout << "Only zero or two arguments are valid" << endl;
	}
}
