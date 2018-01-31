// Required headers (Dependency: opencv, dlib)
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;

// Camera matrix, given focal length and optical center
cv::Mat get_camera_matrix(float focal_length, cv::Point2d center) {
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	return camera_matrix;
}

// 3D model points(world coordinates), http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
std::vector<cv::Point3d> get_3D_model_points(void) {
	std::vector<cv::Point3d> object_pts;
	object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
	object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
	object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
	object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
	object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
	object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
	object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
	object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
	object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
	object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
	object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
	object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
	object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
	object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner	
	return object_pts;
}

int main() {

	// Start camera
	cv::VideoCapture cap(0);

	// If not opened, then exit
	if (!cap.isOpened()) {
		cerr << "Can't open camera" << endl;
		return EXIT_FAILURE;
	}

	// Load face detection model
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor predictor;
	deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

	// Desired size
	int desired_cols = 640, desired_rows = 480;

	// Intrinsic params
	cv::Mat cam_matrix = get_camera_matrix(desired_cols, cv::Point2f(desired_cols / 2, desired_rows / 2));
	
	// Distortion Coefficient (here it is assumed that camera lens is distortion free.
	cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);

	// 3D model points(world coordinates)
	std::vector<cv::Point3d> object_pts = get_3D_model_points();

	// 2D image points(image coordinates), 
	std::vector<cv::Point2d> image_pts;

	// Required matrices to store results
	cv::Mat rotation_vec, translation_vec;          //3 x 1
	cv::Mat rotation_mat;                           //3 x 3 R
	
	cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
	cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

	// 3D world coordinate axis reprojections to verify resulting pose and drawing pose on image
	std::vector<cv::Point3d> reprojectsrc;
	reprojectsrc.push_back(cv::Point3d(0.0, 0.0, 0.0));
	reprojectsrc.push_back(cv::Point3d(-5.0, 0.0, 0.0));
	reprojectsrc.push_back(cv::Point3d(0.0, 5.0, 0.0));
	reprojectsrc.push_back(cv::Point3d(0.0, 0.0, 5.0));

	// Reprojected 2D points
	std::vector<cv::Point2d> reprojectdst;
	reprojectdst.resize(8);

	// Temp buffer for some arguments of function decomposeProjectionMatrix()
	cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
	cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
	cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

	// Displayin text on image
	ostringstream outtext;

	//// Record video
	// cv::VideoWriter outputVideo;
	// outputVideo.open("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 8, 
	//	cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);

	// Main loop
	while (1) {
		// Grab a frame
		cv::Mat temp;
		cap >> temp;

		// If frame is not of desired size, then resize 
		if (temp.rows != desired_rows || temp.cols != desired_cols) {
			cv::resize(temp, temp, cv::Size(desired_cols, desired_rows));
		}

		// Flip in order to display mirror image
		cv::flip(temp, temp, 1);
		cv_image<bgr_pixel> cimg(temp);

		// Detect faces 
		std::vector<rectangle> faces = detector(cimg);

		// Find the pose of each face
		if (faces.size() > 0) {
			
			// Find features, calling required dlib function
			full_object_detection shape = predictor(cimg, faces[0]);

			// Plot feature points
			for (unsigned int i = 0; i < 68; ++i) {
				circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 1, cv::Scalar(255, 255, 255), -1);
				circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(0, 0, 255), 1);
			}

			// 2D reference image points, 
			image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
			image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
			image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
			image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
			image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
			image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
			image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
			image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
			image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
			image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
			image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
			image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
			image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
			image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

			// Calculate pose
			cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

			// Reproject
			cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

			// Calculate euler angles
			cv::Rodrigues(rotation_vec, rotation_mat);
			cv::hconcat(rotation_mat, translation_vec, pose_mat);
			cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
			
			// Roll, Yaw and Pitch
			double roll = euler_angle.at<double>(2), yaw = euler_angle.at<double>(1), pitch = euler_angle.at<double>(0);

			// Print on image           
			outtext << "Roll: " << setprecision(3) << euler_angle.at<double>(2);
			cv::putText(temp, outtext.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
			outtext.str("");
			outtext << "Yaw: " << setprecision(3) << euler_angle.at<double>(1);
			cv::putText(temp, outtext.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
			outtext.str("");
			outtext << "Pitch: " << setprecision(3) << euler_angle.at<double>(0);
			cv::putText(temp, outtext.str(), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
			outtext.str("");

			// Whether face is towards camera or not
			if (roll >= -10 && roll <= 10 && yaw >= -10 && yaw <= 10 && pitch >= -10 && pitch <= 10) {
				outtext << "Towards Camera";
				cv::putText(temp, outtext.str(), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
				outtext.str("");
			}
			else {
				outtext << "Away from camera";
				cv::putText(temp, outtext.str(), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
				outtext.str("");
			}
			image_pts.clear();

			// Draw pose on image
			cv::Point pt;
			pt.x = desired_cols - reprojectdst[0].x - 100;
			pt.y = reprojectdst[0].y - 100;
			reprojectdst[0].x = desired_cols - 100; reprojectdst[0].y = 100;
			reprojectdst[1].x += pt.x; reprojectdst[1].y -= pt.y;
			reprojectdst[2].x += pt.x; reprojectdst[2].y -= pt.y;
			reprojectdst[3].x += pt.x; reprojectdst[3].y -= pt.y;

			cv::line(temp, reprojectdst[0], reprojectdst[1], cv::Scalar(255, 0, 0), 2, 8, 0);
			cv::line(temp, reprojectdst[0], reprojectdst[2], cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::line(temp, reprojectdst[0], reprojectdst[3], cv::Scalar(0, 255, 0), 2, 8, 0);
		}

		//// Write video 
		//outputVideo.write(temp);

		// press escape key to end
		imshow("demo", temp);
		if (cv::waitKey(1) == 27) {
			break;
		}
	}

	return 0;
}
