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
#include <thread>
#include <atomic>
#include "c:/tensorflow1/PnP/PnP/example.hpp"
#include "MarkerDetection.h"

#define MARKER_WIDTH	0.146f
#define MARKER_HEIGHT	0.146f

//////////////////////// 전역 변수 ///////////////////////////
detect detect1;
detect detect2; // detect 구조체 2개 선언
rect rc;
int keyArr[350];
int color_switch = 0;

// pointCloud 변수 선언 //
rs2::pointcloud pc;
rs2::points points;
int memSize; // bounding box crop size


// 함수 선언 //
void register_glfw_callbacks(window& app, glfw_state& app_state);
static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void PrintMarkerInfo(const vector<sMarkerInfo> &markers);

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
    window app(width, height, "Point Cloud"); // detection point cloud
    glfw_state app_state;
    register_glfw_callbacks(app, app_state);
    glfwSetKeyCallback(app.operator GLFWwindow *(), KeyCallback);

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
            detect_num = 1;
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
        }
        if (detect_num == 2) {
            std::cout << "[detect 1: " << detect1.name << " / " << "Prob: " << detect1.score << "]" << std::endl;
            std::cout << "[detect 2: " << detect2.name << " / " << "Prob: " << detect2.score << "]" << std::endl;
        }

        ///////// window 생성 ////////
            // OpenGL 시각화 window //
            // realsense depth data matching //

        pc.map_to(color);
        points = pc.calculate(depth);
        // Key r (Image --> color) : Key t (color --> image) //
        if (color_switch == 0)
            app_state.tex.upload(color);
        if (color_switch == 1)
            app_state.tex.upload(depth_texture); // depth color matching (실 영상이 아닌 뎁스 컬러)
        

        if (detect_num == 1)
            draw_pointcloud(width, height, app_state, points, &detect1);
        /*if (detect_num == 2)
            draw_pointcloud(width, height, app_state, points, &detect1, &detect2);*/

        glPopMatrix();
        glfwSwapBuffers(app.operator GLFWwindow *());
        glfwPollEvents();

        Sleep(1);

        ///////////////// Point Cloud PnP Stage //////////////////
        if (pMapView_signal[0] == 1) {

            // openGL 출력 읽기 //
            //unsigned char *pixels = new unsigned char[sizeof(unsigned char)* (cw1) * (ch1) * 4];
            //cv::Mat pc_image(cv::Size(cw1, ch1), CV_8UC4, pixels, cv::Mat::AUTO_STEP); // pointcloud image capture
            //cv::Mat result; // 상하대칭변환

            //// pointcloud GL Capture --> opencv mat ////
            IplImage *ar = cvCreateImage(cvSize(cw1, ch1), IPL_DEPTH_8U, 3);
            IplImage *mask = cvCreateImage(cvSize(cw1, ch1), IPL_DEPTH_8U, 1);
            IplImage *img_gl = cvCreateImage(cvSize(cw1, ch1), IPL_DEPTH_8U, 3);
            glReadPixels(cx2, cy2, cw1, ch1, GL_BGR_EXT, GL_UNSIGNED_BYTE, img_gl->imageData);
            cvFlip(img_gl, img_gl, -1);
            cvCvtColor(img_gl, mask, CV_RGB2GRAY);
            cvThreshold(mask, mask, 100, 255, CV_THRESH_BINARY);
            cv::Point detect_point;
            detect_point.x = cw1 / 2;
            detect_point.y = 20;
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 1, 1, 0, 2);
            cvPutText(img_gl, detect1.name.c_str(), detect_point, &font, cvScalar(0, 0, 255));
            cvFlip(img_gl, img_gl, 1);

            IplImage *und;
            IplImage *dst;
            IplImage *img_bg;
            

            printf("PointCloud Augmentation Stage\n");
            //glfwDestroyWindow(app.operator GLFWwindow *());
            // dection info 출력 //
            // window 생성 // 

            /// Streaming ///
            cv::Mat cimage; // streaming
            _marker_width = MARKER_WIDTH;
            _marker_height = MARKER_HEIGHT;
            LoadCalibParams(cvSize(1280, 720));

            while (pMapView_signal[0] == 1) {
                //////////// PnP ////////////
                // realsense 영상 stream //
                data = pipe.wait_for_frames();
                color = data.get_color_frame();
                cv::Mat pnp_image(cv::Size(1280, 720), CV_8UC3, (int*)color.get_data(), cv::Mat::AUTO_STEP);
                cv::cvtColor(pnp_image, pnp_image, CV_RGB2BGR);
                img_bg = new IplImage(pnp_image);

                und = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 3);
                dst = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 3);

                // 왜곡 보정 : calibration //
                Undistort(img_bg, und);

                und = new IplImage(und_mat);

                // marker detection //
                cvCopy(img_bg, dst, 0);
                MarkerRecog(und, dst);

                // 인식 마커 출력 //
                PrintMarkerInfo(_markers);
                const float ignoring_margin = 0.f;	// 원본 이미지로부터 마커 이미지로 복사하면서 무시할 테두리의 영역
                // 추출한 마커를 저장할 이미지 상의 좌표
               
                if (!_markers.empty()) {
                    
                    cvCopy(img_gl, ar, mask);
                    
                    cv_ARaugmentImage(img_gl, dst, _markers[0].corner, 2);
                    
                }
                cvShowImage("test", dst);

                // key 'q' is revert to DETECTION MODE //
                int key = cv::waitKey(1);
                if (key == 113) {
                    cv::destroyAllWindows();
                    pMapView_signal[0] = 0;
                    
                }
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

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    keyArr[key] = action;
    //printf("%d", key);
    switch (key) {
        // r: image --> color // t: color --> image;
    case 82:
        color_switch = 1;
        break;

    case 84:
        color_switch = 0;
        break;
        
    }

}

void PrintMarkerInfo(const vector<sMarkerInfo> &markers)
{
    string out_text;

    for (unsigned int i = 0; i < markers.size(); ++i) {
        const sMarkerInfo &mi = markers[i];
        char text[1024];
        sprintf(text,
            "Marker ID          = %d\r\n"
            "Rotation Vector    = %8.3f, %8.3f, %8.3f \r\n"
            "Translation Vector = %8.3f, %8.3f, %8.3f \r\n"
            "\r\n",
            (int)mi.ID,
            (double)mi.rotation[0], (double)mi.rotation[1], (double)mi.rotation[2],
            (double)mi.translation[0], (double)mi.translation[1], (double)mi.translation[2]);

        out_text += text;
    }
}
