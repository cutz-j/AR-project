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

cv::Mat bg_mat;
cv::Mat und_mat;

float _cell_w;	// ü���ǿ��� �� ������ ���ι��� ����
float _cell_h;	// ü���ǿ��� �� ������ ���ι��� ����

int _n_boards;	// �ν��� ü���� ���� �����Ѵ�.
int _board_w;	// ü������ ���ι��� �ڳ� ��
int _board_h;	// ü������ ���ι��� �ڳ� ��
int _board_n;	// ���� x ���� ������ �ڳ� ��
int _board_total;
int _successes;

void LoadCalibParams(CvSize &image_size)
{
     //���Ϸ� ����� ������İ� �ְ� ����� �ҷ�����
    cv::FileStorage fs("Matrix.xml", cv::FileStorage::READ);
    fs["intrinsic"] >> intrinsic_matrix;
    fs["distortion"] >> distortion_coeffs;
    fs.release();

    // �ְ� ���Ÿ� ���� ������ ����
    _mapx = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
    _mapy = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_32F, 1);
    mapx = cv::cvarrToMat(_mapx);
    mapy = cv::cvarrToMat(_mapy);

    // �ְ� ���Ÿ� ���� ������ ����
    cv::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, cv::Mat(), intrinsic_matrix, cv::Size(1280, 720), CV_32FC1, mapx, mapy);

    _successes = _n_boards + 1;
}

void Undistort(IplImage *src, IplImage *dst)
{
    // ī�޶� �Է¿���(src)���� �ְ��� ������ ����(dst)�� �����.
    bg_mat = cv::cvarrToMat(src);
    und_mat = cv::cvarrToMat(dst);

    cv::remap(bg_mat, und_mat, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}
