﻿//#include <iostream>
//// cv //
//#include "opencv2/opencv.hpp"
//#include "MarkerDetection.h"
//#include <librealsense2/rs.hpp>
//
//
//void calibration()
//{
//	rs2::pipeline pipe;
//	rs2::config cfg;
//	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 0);
//	cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_ANY, 0);
//
//	cv::Size boardSize(9, 7);
//	int _board_n = 9 * 7;
//	int _n_boards = 1;
//	int board_w = 9;
//	int board_h = 7;
//	int found;
//
//	std::vector<cv::Point2f> corners;
//	corners.clear();
//	std::vector<cv::Point3f> cornerPoints3D;
//	cornerPoints3D.clear();
//	float square_size = 2.5;
//
//	for (int v = 0; v < 7; v++)
//		for (int u = 0; u < 9; u++) {
//			cv::Point3f tmp;
//			tmp.x = u * square_size;
//			tmp.y = v * square_size;
//			tmp.z = 0;
//			cornerPoints3D.push_back(tmp);
//		}
//
//	IplImage* _mapx;
//	IplImage* _mapy;
//
//	cv::Mat intrinsic_matrix = cv::Mat::zeros(3, 3, CV_64F);
//	cv::Mat distortion_coeffs(8, 1, CV_64F);
//
//	cv::Mat rotationMatrix = cv::Mat::zeros(3, 3, CV_64F);
//	cv::Mat translationVector = cv::Mat::zeros(3, 1, CV_64F);
//
//	pipe.start(cfg);
//
//	while (1) {
//		auto data = pipe.wait_for_frames();
//		auto color = data.get_color_frame();
//		cv::Mat image(cv::Size(1280, 720), CV_8UC3, (int*)color.get_data(), cv::Mat::AUTO_STEP);
//		cv::cvtColor(image, image, CV_RGB2BGR);
//
//		IplImage *dst = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 3);
//		IplImage *gray = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 1);
//
//		cv::Mat dst_mat = cv::cvarrToMat(dst);
//		
		//found = cv::findChessboardCorners(image, boardSize, corners);

		////// 코너를 dst 이미지에 그린다.
  //      if (found)
		//    cv::drawChessboardCorners(image, boardSize, cv::Mat(corners), found);
//		
//		// 코너 데이터 저장 //
//		cv::imshow("window", image);
//		cvWaitKey(1);
//
//		if (found)
//			break;
//
//	}
//
//	std::vector<std::vector<cv::Point2f>> imagePoints;
//	imagePoints.push_back(corners);
//
//	std::vector<std::vector<cv::Point3f>> objPoints;
//	objPoints.push_back(cornerPoints3D);
//
//	cv::Size imageSize = cv::Size(1280, 720);
//
	// 컬리브레이션 //
	//cv::calibrateCamera(objPoints, imagePoints, imageSize, intrinsic_matrix, distortion_coeffs, rotationMatrix, translationVector);

	//// 파일 저장 //
	//cv::FileStorage fs1("Matrix.xml", cv::FileStorage::WRITE);
 //   fs1 << "intrinsic" << intrinsic_matrix;
 //   fs1 << "distorion" << distortion_coeffs;
	//fs1.release();
//
//
//	_mapx = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
//	_mapy = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
//	cv::Mat mapx = cv::cvarrToMat(_mapx);
//	cv::Mat mapy = cv::cvarrToMat(_mapy);
//
//	// 왜곡 제거를 위한 지도를 구성
//	cv::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, cv::Mat(), intrinsic_matrix, cv::Size(1280, 720), CV_32FC1, mapx, mapy);
//	
//}
