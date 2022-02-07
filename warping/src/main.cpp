#include <iostream>
//#include <experimental/filesystem>
#include <filesystem>
#include <string>
#include <map>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any libigl headers here ***/
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

#include "warping.h"

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;
using namespace Eigen;
namespace fs = std::__fs::filesystem;

std::map<int, int> landmarks_template, landmarks_target;
// Vertex array, #V x3
Eigen::MatrixXd V_original, V_start, V_target;
// Face array, #F x3
Eigen::MatrixXi F_start, F_target;

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        V_start = computeWarping(V_start, F_start, V_target, landmarks_template, landmarks_target);
        viewer.data().clear();
        viewer.data().set_mesh(V_start, F_start);
    }
    if(key == '2'){
        viewer.data().clear();
        viewer.data().set_mesh(V_original, F_start);
    }
    if(key == '3'){
        viewer.data().clear();
        viewer.data().set_mesh(V_start, F_start);
    }
    if(key == '4'){
        viewer.data().clear();
        viewer.data().set_mesh(V_target, F_target);
    }

    return true;
}

map<int, int> get_landmarks(string path){
    map<int, int> out;
    ifstream infile(path.c_str());
    std::string line;
    int a, b;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        if (!(iss >> a >> b)) { break; } // error
        out[b] = a;
    }
    return out;
}

void compute_example(){
    // Show the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    string filename_start = std::string("../data/template_rigid_aligned_scaled.obj");
    string filename_target = std::string("../data/person0__rigid_aligned_scaled0p01.obj");
    igl::readOBJ(filename_start,V_start,F_start);
    igl::readOBJ(filename_target,V_target,F_target);
    V_original = V_start;

    string landmark_str = "../data/headtemplate_23landmarks";
    landmarks_template = get_landmarks(landmark_str);

    landmark_str = "../data/person0__23landmarks";
    landmarks_target = get_landmarks(landmark_str);

//    callback_key_down(viewer, '1', 0);

    viewer.launch();
}

void compute_folder(string folder_in, string folder_out) {
    fs::path path_in(folder_in);
    fs::path path_out(folder_out);
    string template_output_suffix = "_template.obj";
    //{scan, template}, {scan_landmarks, template_landmarks}
    vector<pair<pair<string, string>, pair<string, string>>> items;
    vector<string> items_out;
    for (auto &p: fs::directory_iterator(path_in)){
        if (p.is_regular_file()) {
            string filename = p.path().filename().string();
//            std::cout << filename << '\n';
            if(filename.ends_with("_template.obj") || filename.ends_with(".txt") || filename.starts_with(".")){
                continue;
            }
//            std::cout << p.path() << '\n';
            string scan_path = p.path().string();
            string template_path = p.path().parent_path().string() + "/" + p.path().stem().string() + template_output_suffix;
            string scan_landmarks = p.path().parent_path().string() + "/" + p.path().stem().string() + ".txt";
//            string template_landmarks = "/Users/msladek/Documents/eth/shape/smgp-3D-face-modeling-and-learning/rigid_align/data/template/headtemplate_noneck.txt";
            string template_landmarks = path_in.string() + "/" + "orig_template/headtemplate_4k_landmarks.txt";
            pair<string, string> first_part = {scan_path, template_path};
            pair<string, string> second_part = {scan_landmarks, template_landmarks};
            items.push_back({first_part, second_part});
            string path_out_mesh = path_out.string() + "/" + p.path().filename().string();
            items_out.push_back(path_out_mesh);
        }
    }
    for (int i = 0;i<items.size();i++) {
        string scan_path = items[i].first.first;
        string template_path = items[i].first.second;
        string scan_landmarks = items[i].second.first;
        string template_landmarks = items[i].second.second;
        igl::readOBJ(template_path,V_start,F_start);
        igl::readOBJ(scan_path,V_target,F_target);
        landmarks_template = get_landmarks(template_landmarks);
        landmarks_target = get_landmarks(scan_landmarks);
        MatrixXd V_output = computeWarping(V_start, F_start, V_target, landmarks_template, landmarks_target);
        string output_path = items_out[i];
        igl::writeOBJ(output_path, V_output, F_start);
        cout << "done " << i+1 << "/" << items.size() << endl;
    }
}

int main(int argc, char *argv[]) {
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
