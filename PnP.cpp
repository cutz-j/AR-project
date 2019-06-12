// RealSense //
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
#include "example.hpp"

struct detect {
    // detect 정보를 저장하는 구조체 //
    float x_min, x_max, y_min, y_max;
    float score;
    std::string name;
};

int row = 40;
int col = 7;
int size = row * col;
detect detect1, detect2; // detect 구조체 2개 선언

// pointCloud 변수 선언 //
rs2::pointcloud pc;
rs2::points points;
int memSize; // bounding box crop size
unsigned char *p_image;
float *xyz;
unsigned char *rgb;
GLfloat TranslateX;
GLfloat TranslateY;
GLfloat TranslateZ;
GLfloat RotateX;
GLfloat RotateY;
GLfloat RotateZ;
GLfloat ZoomScale;
int w = 640, h = 480;
float mousex = 0, mousey = 0;
bool dragging = false;
int pressing = 0;
int keyArr[350];
GLFWwindow* window;
float depth_scale;

// 함수 선언 //
static void Initialize(void);
void renderPointClouds();
static void RenderScene(GLFWwindow* window);
static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void MouseClickCallback(GLFWwindow* window, int button, int action, int mods);
static void MouseMotionCallback(GLFWwindow* window, double x, double y);
static void MouseScrollCallback(GLFWwindow* window, double dx, double dy);


int main(int argc, char * argv[]) try
{
    // 변수 선언 //
	HANDLE hMapFile_image;
	HANDLE hMapFile_info;
	HANDLE hMapFile_signal;
	int *pMapView_image;
	float *pMapView_info;
	int *pMapView_signal;
	int class_num;
	int size;
    int detect_num;

	rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);
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
	hMapFile_info = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(float), L"INFO");
	hMapFile_signal = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(int), L"SIGNAL");
    pMapView_signal = (int*)MapViewOfFile(hMapFile_signal, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    pMapView_signal[0] = 0;

    texture depth_image, color_image;
   
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
                detect1.y_min = pMapView_info[0 + (i * 7)];
                detect1.x_min = pMapView_info[1 + (i * 7)];
                detect1.y_max = pMapView_info[2 + (i * 7)];
                detect1.x_max = pMapView_info[3 + (i * 7)];
                detect1.score = pMapView_info[4 + (i * 7)];
                class_num = (int)pMapView_info[5 + (i * 7)];
                detect1.name = classes[class_num - 1];
            }
            if (i == 1) {
                detect2.y_min = pMapView_info[0 + (i * 7)];
                detect2.x_min = pMapView_info[1 + (i * 7)];
                detect2.y_max = pMapView_info[2 + (i * 7)];
                detect2.x_max = pMapView_info[3 + (i * 7)];
                detect2.score = pMapView_info[4 + (i * 7)];
                class_num = (int)pMapView_info[5 + (i * 7)];
                detect2.name = classes[class_num - 1];
            }
		}

		///////////////// Point Cloud Stage //////////////////
		if (pMapView_signal[0] == 1) {
			printf("PointCloud Stage\n");

            // dection info 출력 //
            if (detect_num == 1)
                std::cout << "[detect 1: " << detect1.name << " / " << "Prob: " << detect1.score << "]" << std::endl;
            if (detect_num == 2)
                std::cout << "[detect 1: " << detect1.name << " / " << "Prob: " << detect1.score << "]" << std::endl;
                std::cout << "[detect 2: " << detect2.name << " / " << "Prob: " << detect2.score << "]" << std::endl;

            // OpenGL 시각화 window //
           
            glfwInit();
            window = glfwCreateWindow(width, height, "Point Cloud", NULL, NULL);
            glfwMakeContextCurrent(window);
            memSize = w * h;

            Initialize(); // init
            // realsense depth data matching //
            pc.map_to(depth);
            points = pc.calculate(depth); // depthmap에서 pointcloud 계산
            auto vertice = points.get_vertices();

            for (int i = 0; i < memSize; i++) {
                rgb[i * 3] = (int)p_image[i * 3];
                rgb[i * 3 + 1] = (int)p_image[i * 3 + 1];
                rgb[i * 3 + 2] = (int)p_image[i * 3 + 2];
                xyz[i * 3] = (float)vertice[i].x;
                xyz[i * 3 + 1] = (float)vertice[i].y;
                xyz[i * 3 + 2] = (float)vertice[i].z;
            }


            glfwSetKeyCallback(window, KeyCallback);
            glfwSetMouseButtonCallback(window, MouseClickCallback);
            glfwSetCursorPosCallback(window, MouseMotionCallback);
            glfwSetScrollCallback(window, MouseScrollCallback);

            
			while (pMapView_signal[0] == 1) {
                //////////// Rendering ////////////
                RenderScene(window);
                glfwSwapBuffers(window);
                glfwPollEvents();

            }
		    }
        glfwDestroyWindow(window);
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

static void Initialize(void) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glClearColor(0.0, 0.0, 0.0, 1.0); 
  
    TranslateX = 0;
    TranslateY = 0;
    TranslateZ = 0;
    RotateX = 0;
    RotateY = 0;
    RotateZ = 0;
    ZoomScale = 1.0;

    xyz = new float[memSize * 3];
    rgb = new unsigned char[memSize * 3];
    memset(rgb, 0, sizeof(unsigned char) * memSize * 3);
    memset(xyz, 0, sizeof(float) * memSize * 3);
}

void renderPointClouds() {
    glPointSize(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
    glBegin(GL_POINTS);
    
    auto tex_coords = points.get_texture_coordinates(); // and texture coordinates

    for (int i = 0; i < memSize; i++) {
        if (xyz[i * 3 + 2]) {
            glColor3ub(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            glVertex3f(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
            glTexCoord2fv(tex_coords[i]);
        }
    }

    glEnd();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}

static void RenderScene(GLFWwindow* window) {
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, 640 / 480, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(TranslateX, TranslateY, TranslateZ);
    glRotatef(RotateX, 1.0f, 0.0f, 0.0f);
    glRotatef(RotateY, 0.0f, 1.0f, 0.0f);
    glRotatef(RotateZ, 0.0f, 0.0f, 1.0f);
    glScalef(ZoomScale, ZoomScale, ZoomScale);
    glTranslatef(0, 0, -0.5f);

    renderPointClouds();
    glPopMatrix();
    
}

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    keyArr[key] = action;
    switch (key) {
    case 262:
        TranslateX += 0.01;
        break;
    case 263:
        TranslateX -= 0.01;
        break;
    case 264:
        TranslateY -= 0.01;
    case 265:
        TranslateY += 0.01;
        break;
    }

}

static void MouseClickCallback(GLFWwindow* window, int button, int action, int mods) {
    switch (button) {
    case GLFW_MOUSE_BUTTON_1:
        dragging = action;
        break;
    }
}

static void MouseMotionCallback(GLFWwindow* window, double x, double y) {
    double scale = 0.1;
    if (dragging) {
        RotateY += (mousex - x) * scale;
        RotateX += (mousey - y) * scale;
    }

    mousex = x;
    mousey = y;
}

static void MouseScrollCallback(GLFWwindow* window, double dx, double dy) {
    ZoomScale += dy;
}
