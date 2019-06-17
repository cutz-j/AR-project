#pragma once


void LoadCalibParams (CvSize &image_size);
void Undistort(IplImage *src, IplImage *dst);
	
CvMat* _image_points;
CvMat* _object_points;
CvMat* _point_counts;

cv::Mat intrinsic_matrix;
cv::Mat distortion_coeffs;

IplImage* _mapx;
IplImage* _mapy;
cv::Mat mapx;
cv::Mat mapy;

float _cell_w;	// 체스판에서 한 격자의 가로방향 넓이
float _cell_h;	// 체스판에서 한 격자의 세로방향 넓이

int _n_boards;	// 인식할 체스판 수를 지정한다.
int _board_w;	// 체스판의 가로방향 코너 수
int _board_h;	// 체스판의 세로방향 코너 수
int _board_n;	// 가로 x 세로 방향의 코너 수
int _board_total;
int _successes;

void LoadCalibParams(CvSize &image_size)
{
     //파일로 저장된 내부행렬과 왜곡 계수를 불러오기
    cv::FileStorage fs("Matrix.xml", cv::FileStorage::READ);
    fs["intrinsic"] >> intrinsic_matrix;
    fs["distcoeffs"] >> distortion_coeffs;
    fs.release();

    // 왜곡 제거를 위한 지도를 생성
    _mapx = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
    _mapy = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
    mapx = cv::cvarrToMat(_mapx);
    mapy = cv::cvarrToMat(_mapy);

    // 왜곡 제거를 위한 지도를 구성
    cv::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, cv::Mat(), intrinsic_matrix, cv::Size(1280, 720), CV_32FC1, mapx, mapy);

    _successes = _n_boards + 1;
}

void Undistort(IplImage *src, IplImage *dst)
{
    // 카메라 입력영상(src)에서 왜곡을 제거한 영상(dst)을 만든다.
    cv::Mat src_mat = cv::cvarrToMat(src);
    cv::Mat dst_mat = cv::cvarrToMat(dst);

    cv::remap(src_mat, dst_mat, mapx, mapy, 1);
}

