// RealSense //
#include <iostream>
#include <librealsense2/rs.hpp>
#include <algorithm>
#include "example.hpp"

// memory map //
#include <conio.h>
#include <Windows.h>

// cv //
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// lib //
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <functional>
#include "GLFW/glfw3.h"

void register_glfw_callbacks(window& app, glfw_state& app_state);
void draw_pointcloud(float width, float height, glfw_state& app_state, rs2::points& points);
int row = 40;
int col = 7;
int size = row * col;

int main(int argc, char * argv[]) try
{
	HANDLE hMapFile_image;
	HANDLE hMapFile_info;
	int *pMapView_image;
	float *pMapView_info;
	float detect_num;
	float x, y, w, h;
	float score;
	int class_num;
	unsigned char *p_image;
	int size;

	window app(640, 480, "RealSense Pointcloud Example");
	glfw_state app_state;
	register_glfw_callbacks(app, app_state);

	rs2::pointcloud pc;
	rs2::points points;
	rs2::pipeline pipe;

	pipe.start();

	rs2::frameset data = pipe.wait_for_frames();
	rs2::frame color = data.get_color_frame();

	const int width = color.as<rs2::video_frame>().get_width();
	const int height = color.as<rs2::video_frame>().get_height();

	size = width * height * 3;

	hMapFile_image = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(int) * size, L"IMAGE");
	hMapFile_info = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(float), L"INFO");


	while (1) {

		// RealSense API READ stage //
		rs2::frameset data = pipe.wait_for_frames();
		rs2::frame color = data.get_color_frame();
		rs2::frame depth = data.get_depth_frame();

		cv::Mat image(cv::Size(width, height), CV_8UC3, (int*)color.get_data(), cv::Mat::AUTO_STEP);
		p_image = image.data;

		pMapView_image = (int*)MapViewOfFile(hMapFile_image, FILE_MAP_ALL_ACCESS, 0, 0, 0);
		for (int a = 0; a < size; a++)
			pMapView_image[a] = (int)p_image[a];
		
		pMapView_info = (float*)MapViewOfFile(hMapFile_info, FILE_MAP_ALL_ACCESS, 0, 0, 0);
		

		if (!color)
			color = data.get_infrared_frame();

		points = pc.calculate(depth); // generate the pointcloud and texture mappings
		pc.map_to(color);
		app_state.tex.upload(color); // upload the color frame to OpenGL

		detect_num = pMapView_info[6];
		for (int i = 0; i < detect_num; i++) {
			x = pMapView_info[0 + (i * 7)];
			y = pMapView_info[1 + (i * 7)];
			w = pMapView_info[2 + (i * 7)];
			h = pMapView_info[3 + (i * 7)];
			score = pMapView_info[4 + (i * 7)];
			class_num = (int)pMapView_info[5 + (i * 7)];
			printf("[%f %f %f %f %f %d] \n", x, y, w, h, score, class_num);
		}
	}	

	_getch();
	UnmapViewOfFile(hMapFile_image);
	UnmapViewOfFile(hMapFile_info);

	return 0;
}
catch (const rs2::error & e) {
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e) {
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}