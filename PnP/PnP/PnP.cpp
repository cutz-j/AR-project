﻿// RealSense //
#include <iostream>
#include <librealsense2/rs.hpp>
#include <algorithm>


// OpenGL //
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GLFW/glfw3.h>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glfw3dll.lib")

// memory map //
#include <conio.h>
#include <Windows.h>
#include "C:/tensorflow1/PnP/PnP/class.hpp" // class array

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
#include "c:/tensorflow1/PnP/PnP/example.hpp"
#include <thread>
#include <atomic>

//////////////////////// 전역 변수 ///////////////////////////
detect detect1;
detect detect2; // detect 구조체 2개 선언
rect rc;

// pointCloud 변수 선언 //
rs2::pointcloud pc;
rs2::points points;
int memSize; // bounding box crop size

int x, y, w1, h1, w2, h2;

// 함수 선언 //
void register_glfw_callbacks(window& app, glfw_state& app_state);


int main(int argc, char * argv[]) try
{
    ///////////// 변수 선언 /////////////////
    HANDLE hMapFile_image;
    HANDLE hMapFile_info;
    HANDLE hMapFile_signal;
    int *pMapView_image;
    float *pMapView_info;
    int *pMapView_signal;

    // box 정보 //
    int class_num, size, detect_num;
    int row = 40;
    int col = 7;
    int size_info = row * col;
    unsigned char *p_image;

    // realsense //
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 0);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_ANY, 0);
    rs2::colorizer colorizer;
    rs2::frame  depth_texture;

    pipe.start(cfg); // realsense pipeline start

    // 초기 데이터 size 잡기 //
    auto data = pipe.wait_for_frames();
    auto color = data.get_color_frame();
    auto depth = data.get_depth_frame();

    
    const int width = color.as<rs2::video_frame>().get_width();
    const int height = color.as<rs2::video_frame>().get_height();

    size = width * height * 3;

    // 공유메모리 초기화 //
    hMapFile_image = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(int) * size, L"IMAGE");
    hMapFile_info = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(float) * size_info, L"INFO");
    hMapFile_signal = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(int), L"SIGNAL");
    pMapView_signal = (int*)MapViewOfFile(hMapFile_signal, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    pMapView_signal[0] = 0;


    // GL window Init //
    window app(width, height, "Point Cloud1"); // detection point cloud
    glfw_state app_state;
    register_glfw_callbacks(app, app_state);
  
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        MessageBox(NULL, TEXT("FreeType 라이브러리 초기화 실패"), TEXT("에러"), MB_OK);
        return -1;
    }

    FT_Face face;
    if (FT_New_Face(ft, "FreeSans.ttf", 0, &face)) {
        fprintf(stderr, "Could not open font\n");
        return 1;
    }

    FT_Set_Pixel_Sizes(face, 0, 32);
  
    if (FT_Load_Char(face, 'Z', FT_LOAD_RENDER)) {
        fprintf(stderr, "Could not load character 'Z'\n");
        return 1;
    }

    FT_GlyphSlot g = face->glyph;
    //attribute vec4 coord;
    //varying vec2 texcoord;



    // 구동 시작 //
    while (1) {
        // RealSense API READ stage //
        data = pipe.wait_for_frames();
        color = data.get_color_frame();
        depth = data.get_depth_frame(); // depth frame
        depth_texture = colorizer.colorize(depth);

        // image 전환 후, 공유메모리 전송 //
        cv::Mat image(cv::Size(width, height), CV_8UC3, (int*)color.get_data(), cv::Mat::AUTO_STEP);
        p_image = image.data;
        rgb = image.data; // 시각화 color image;
        pMapView_image = (int*)MapViewOfFile(hMapFile_image, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        for (int a = 0; a < size; a++)
            pMapView_image[a] = (int)p_image[a];

        // detection / signal(PointCloud 전환) 공유 메모리 //
        pMapView_info = (float*)MapViewOfFile(hMapFile_info, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        pMapView_signal = (int*)MapViewOfFile(hMapFile_signal, FILE_MAP_ALL_ACCESS, 0, 0, 0);

        ///// bounding box information receive /////
        // detect num 2개를 받아 구조체에 저장 //
        detect_num = pMapView_info[6]; // detect된 수 가져오기
        // detect num은 2개로 제한 --> 3개 이상일 시에는 2개만 가져와서 2개의 구조체에 저장 //
        if (detect_num > 2)
            detect_num = 2;
        for (int i = 0; i < detect_num; i++) {
            if (i == 0) {
                detect1.y_min = (int)pMapView_info[0 + (i * 7)];
                detect1.x_min = (int)pMapView_info[1 + (i * 7)];
                detect1.y_max = (int)pMapView_info[2 + (i * 7)];
                detect1.x_max = (int)pMapView_info[3 + (i * 7)];
                detect1.score = pMapView_info[4 + (i * 7)];
                class_num = (int)pMapView_info[5 + (i * 7)];
                detect1.name = classes[class_num - 1];
            }
            if (i == 1) {
                detect2.y_min = (int)pMapView_info[0 + (i * 7)];
                detect2.x_min = (int)pMapView_info[1 + (i * 7)];
                detect2.y_max = (int)pMapView_info[2 + (i * 7)];
                detect2.x_max = (int)pMapView_info[3 + (i * 7)];
                detect2.score = pMapView_info[4 + (i * 7)];
                class_num = (int)pMapView_info[5 + (i * 7)];
                detect2.name = classes[class_num - 1];
            }
        }

        if (detect_num == 1) {
            std::cout << "[detect 1: " << detect1.name << " / " << "Prob: " << detect1.score << "]" << std::endl;
            w1 = detect1.x_max - detect1.x_min + 10;
            h1 = detect1.y_max - detect1.y_min + 10;
        }
        if (detect_num == 2) {
            std::cout << "[detect 1: " << detect1.name << " / " << "Prob: " << detect1.score << "]" << std::endl;
            std::cout << "[detect 2: " << detect2.name << " / " << "Prob: " << detect2.score << "]" << std::endl;
            w1 = detect1.x_max - detect1.x_min + 10;
            h1 = detect1.y_max - detect1.y_min + 10;
            w2 = detect2.x_max - detect2.x_min + 10;
            h2 = detect2.y_max - detect2.y_min + 10;
        }

        ///////// window 생성 ////////
            // OpenGL 시각화 window //
            // realsense depth data matching //

        pc.map_to(color);
        points = pc.calculate(depth);
        app_state.tex.upload(color);
        //app_state.tex.upload(depth_texture); // depth color matching (실 영상이 아닌 뎁스 컬러)
        

        if (detect_num == 1)
            draw_pointcloud(width, height, app_state, points, &detect1);
        if (detect_num == 2)
            draw_pointcloud(width, height, app_state, points, &detect1, &detect2);

        glPopMatrix();
        glfwSwapBuffers(app.operator GLFWwindow *());
        glfwPollEvents();


        Sleep(1);

        ///////////////// Point Cloud PnP Stage //////////////////
        if (pMapView_signal[0] == 1) {
            printf("PointCloud PnP Stage\n");

            // dection info 출력 //
            // window 생성 //
            

            
            
           
            while (pMapView_signal[0] == 1) {
                //////////// Rendering ////////////
                
            }
        }
    }
    
	_getch();
	UnmapViewOfFile(hMapFile_image);
	UnmapViewOfFile(hMapFile_info);
    UnmapViewOfFile(hMapFile_signal);

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
