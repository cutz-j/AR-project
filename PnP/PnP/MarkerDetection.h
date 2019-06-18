#ifndef __MARKER_DETECTION_H__
#define __MARKER_DETECTION_H__ 

#include <vector>
#include <opencv/highgui.h>
#include "CamCalib.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;

struct sMarkerInfo 
{
	int level;
	float width, height;	// ��Ŀ�� ũ��
	CvPoint2D32f center;	// ��Ŀ�� �߽� ��
	CvPoint2D32f corner[4];	// �簢���� �� �ڳ� ��
	int ID;					// ��Ŀ���� �νĵ� ID
	
	float rotation[3];		// ��Ŀ�� ȸ���� ��Ÿ���� ���� (3 x 1)
	float translation[3];	// ��Ŀ�� �̵��� ��Ÿ���� ���� (3 x 1)
};

vector<sMarkerInfo> _markers;
float _marker_width;
float _marker_height;
CvFont _font;
CvMat *transform_matrix;
const double CV_AR_DISP_SCALE_FIT = 0.0;
const int CV_AR_MARKER_SIZE = 200;


void FindMarkerInContour (CvSeq *contours, CvMemStorage *storage, int level);
bool CheckRectCenter(CvSeq *seq);
void GetMarkerCode(IplImage *src, IplImage *dst);
void ExtractMarkerImage (IplImage *src, IplImage *dst, sMarkerInfo &mi);
void ExtractCodeFromImage (IplImage *src, double code_matrix[6][6]);
bool CheckParity (double code_matrix[6][6]);
int  GetRotation (double code_matrix[6][6]);
void RotateMatrix (double code_matrix[6][6], int rotate_index);
void RotateCorner (CvPoint2D32f corner[4], int angle_idx, int dir);
int  CalcMarkerID (double code_matrix[6][6]);
void FindMarkerPos3d (sMarkerInfo *marker);
void DrawMarkerInfo (sMarkerInfo *marker, IplImage *dst);
void ShowMarkerCode (CvSize &size, double code_matrix[6][6]);
void cv_ARaugmentImage(IplImage* display, IplImage* img, CvPoint2D32f srcQuad[4], double scale);



void DrawMarkerRect(IplImage *img, sMarkerInfo &mi, CvScalar color)
{
    CvPoint corner[4];

    for (int i = 0; i < 4; ++i) {
        corner[i] = cvPointFrom32f(mi.corner[i]);
    }

    cvLine(img, corner[0], corner[1], color, 2);
    cvLine(img, corner[1], corner[2], color, 2);
    cvLine(img, corner[2], corner[3], color, 2);
    cvLine(img, corner[3], corner[0], color, 2);

    cvLine(img, corner[0], corner[2], color, 2);
    cvLine(img, corner[1], corner[3], color, 2);
}

void MarkerRecog(IplImage *src, IplImage *dst)
{

    IplImage *img_gray = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 1);
    IplImage *img_bin = cvCreateImage(cvSize(1280, 720), IPL_DEPTH_8U, 1);

    // �Է��̹����� gray �̹����� �ٲ۴�.
    cvCvtColor(src, img_gray, CV_RGB2GRAY);

    // ����� �����ϱ� ���Ͽ� ����þ� Ŀ���� �����Ͽ� �̹����� �ε巴�� �����.
    // cvSmooth (img_gray, img_gray, CV_GAUSSIAN, 3, 3);

    // �������� ������ �����Ͽ� �̹����� ���� �κ��� ���ش�.
    // IplConvKernel *kernel = cvCreateStructuringElementEx (3, 3, 1, 1, CV_SHAPE_ELLIPSE, NULL);
    // cvMorphologyEx(img_gray, img_gray, NULL, kernel, CV_MOP_CLOSE, 1);
    // cvReleaseStructuringElement (&kernel);

    // gray �̹����� �����Ͽ� threshold���� �������� binary �̹����� �����.
    cvCopy(img_gray, img_bin);

    // �� �Լ� �� �ϳ��� ��� ����.
    //cvThreshold (img_bin, img_bin, 63, 255, CV_THRESH_BINARY /* | CV_THRESH_OTSU*/);
    cvAdaptiveThreshold(img_bin, img_bin, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 31, 15);

    // For debugging
     //cvShowImage ("img_bin", img_bin);

    // ����� ã�� ������ �޸� ���� �Ҵ�
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvSeq *contours = NULL;

    int noContour = cvFindContours(img_bin, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    // int noContour = cvFindContours (img_bin, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    if (0 < noContour) {
        CvSeq *approxContours = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 1., 1);
        // cvDrawContours (dst, approxContours, CV_RGB(255,255,0), CV_RGB(0,255,0), 10, 1, CV_AA);

        _markers.clear();
        FindMarkerInContour(approxContours, storage, 0);
        GetMarkerCode(img_gray, dst);

        vector<sMarkerInfo> markers_tmp;
        for (unsigned int i = 0; i < _markers.size(); i++) {
            if (0 <= _markers[i].ID) {
                markers_tmp.push_back(_markers[i]);
            }
        }
        _markers.swap(markers_tmp);

        //for (unsigned int i=0; i<_markers.size(); ++i) {
        //	DrawMarkerRect(dst, _markers[i], CV_RGB (255, 0, 0));
        //}
    }
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&img_bin);
    cvReleaseImage(&img_gray);
}

inline double Distance(CvPoint &p1, CvPoint &p2)
{
    // �� �� p1, p2���� ��Ŭ���� �Ÿ��� ����Ѵ�.

    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;

    return sqrt(dx*dx + dy * dy);
}

bool CheckRectCenter(CvSeq *seq)
{
    CvPoint corner[4] = {
        *(CvPoint *)cvGetSeqElem(seq, 0),
        *(CvPoint *)cvGetSeqElem(seq, 1),
        *(CvPoint *)cvGetSeqElem(seq, 2),
        *(CvPoint *)cvGetSeqElem(seq, 3),
    };

    // �� �簢���� �� �밢�� ������ �簢�� �߾ӿ� ������ �˻��Ѵ�.
    // ���⼭�� �� ���� l1, l2�� �������� t�� u ���� ����ϴ� ���̴�.
    // l1 = (a1,b1) + t*(x1,y1)
    // l2 = (a2,b2) + u*(x2,y2)

    double a1 = corner[0].x;
    double b1 = corner[0].y;
    double x1 = corner[2].x - corner[0].x;
    double y1 = corner[2].y - corner[0].y;

    double a2 = corner[1].x;
    double b2 = corner[1].y;
    double x2 = corner[3].x - corner[1].x;
    double y2 = corner[3].y - corner[1].y;

    // ���� ���� ������ ���� ����: ������ �����İ� ����.docx
    CvMat *A = cvCreateMat(2, 2, CV_64FC1);
    CvMat *B = cvCreateMat(2, 1, CV_64FC1);
    CvMat *Ainv = cvCreateMat(2, 2, CV_64FC1);
    CvMat *x = cvCreateMat(2, 1, CV_64FC1);

    cvmSet(A, 0, 0, 2 * (x1*x1 + y1 * y1));
    cvmSet(A, 0, 1, -2 * (x1*x2 + y1 * y2));
    cvmSet(A, 1, 0, -2 * (x1*x2 + y1 * y2));
    cvmSet(A, 1, 1, 2 * (x2*x2 + y2 * y2));

    cvmSet(B, 0, 0, 2 * (x1*(a2 - a1) + y1 * (b2 - b1)));
    cvmSet(B, 1, 0, -2 * (x2*(a2 - a1) + y2 * (b2 - b1)));

    cvInvert(A, Ainv);
    cvMatMul(Ainv, B, x);

    double x00 = cvmGet(x, 0, 0);
    double x10 = cvmGet(x, 1, 0);

    cvReleaseMat(&A);
    cvReleaseMat(&B);
    cvReleaseMat(&Ainv);
    cvReleaseMat(&x);

    const double l_th = 0.15;
    if (fabs(x00 - 0.5) < l_th && fabs(x10 - 0.5) < l_th) {
        // ����
        return true;
    }
    else return false;

    /*
    MatrixXd A(2,2);
    A(0,0) =  2*(x1*x1 + y1*y1);
    A(0,1) = -2*(x1*x2 + y1*y2);
    A(1,0) = -2*(x1*x2 + y1*y2);
    A(1,1) =  2*(x2*x2 + y2*y2);

    MatrixXd B(2,1);
    B(0,0) =  2*(x1*(a2 - a1) + y1*(b2 - b1));
    B(1,0) = -2*(x2*(a2 - a1) + y2*(b2 - b1));

    // t, u ���
    MatrixXd x = A.inverse()*B;

    // t, u ���� 0.5�� �������� +-0.15 ���̿� ������ �˻��Ѵ�.
    const double l_th = 0.15;
    if (fabs(x(0,0) - 0.5) < l_th && fabs(x(1,0) - 0.5) < l_th) {
        // ����
        return true;
    }
    else return false;
    */
}

void FindMarkerInContour(CvSeq *contours, CvMemStorage *storage, int level)
{
    for (CvSeq *s = contours; s; s = s->h_next) {
        // ����� �����ϴ� ���� ���� 4�� �̻� �Ǿ�� �簢�� �ĺ��� �ȴ�.
        if (s->total >= 4) {
            // �ٿ�� �ڽ��� ã�� ������ �������� �뷫���� ũ�⸦ �˱� ���ؼ���.
            // ũ�⿡ ���� ����� approximation �ϴ� ���е��� �����Ѵ�.
            // ���⼭�� �뷫 10%������ ���е��� �����Ѵ�. (d*approx_param �κ�)
            CvRect rect = cvBoundingRect(s);

            double d = sqrt((double)rect.height*rect.width);

            const double d_th = 12.;
            const double approx_param = 0.1;

            // �������� �뷫���� ũ�Ⱑ d_th���� Ŀ�� �Ѵ�.
            if (d > d_th) {
                CvSeq *ss = cvApproxPoly(s, s->header_size, storage, CV_POLY_APPROX_DP, d*approx_param, 0);
                // ����� approximation �ϰ��� �ڳ��� ���� 4��(�簢��)���� �˻��Ѵ�.
                if (ss->total == 4) {
                    // �߰�������, �� �簢���� �� �밢�� ������ �簢�� �߾ӿ� ������ �˻��Ѵ�.
                    if (CheckRectCenter(ss)) {
                        // ��Ŀ�� ã�Ҵ�. ��Ŀ ��Ͽ� �����Ѵ�.
                        sMarkerInfo mi;

                        mi.level = level;
                        mi.width = _marker_width;		// ���� ��Ŀ�� ���� ���� (����: m)
                        mi.height = _marker_height;		// ���� ��Ŀ�� ���� ���� (����: m)
                        mi.ID = -1;						// -1�� �ʱ�ȭ
                        mi.corner[0] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 0));
                        mi.corner[1] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 1));
                        mi.corner[2] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 2));
                        mi.corner[3] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 3));

                        _markers.push_back(mi);
                    }
                }
            }
        }

        if (s->v_next) {
            FindMarkerInContour(s->v_next, storage, level + 1);
        }
    }
}

void GetMarkerCode(IplImage *src, IplImage *dst)
{
    for (unsigned int i = 0; i < _markers.size(); ++i) {
        // ����� ��Ŀ�� �ڳʷκ��� �����ȼ� ��Ȯ���� �ڳ� ��ǥ�� �ٽ� ���Ѵ�.
        //cvFindCornerSubPix(src, _markers[i].corner, 4, cvSize(2, 2), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.01));

        // src���� ã�� ��Ŀ�� �������κ��� ��Ŀ ������ �����Ѵ�.
        const int marker_size = 60;
        IplImage *img_marker = cvCreateImage(cvSize(marker_size, marker_size), IPL_DEPTH_8U, 1);
        ExtractMarkerImage(src, img_marker, _markers[i]);

        // ��Ŀ�� 6 x 6�� ��ķ� �����ȴ�.
        double code_matrix[6][6] = { 0, };

        // ��Ŀ ������ �ȼ����� �����κ��� �ڵ� ���� �����Ѵ�.
        ExtractCodeFromImage(img_marker, code_matrix);

        if (CheckParity(code_matrix)) {
            int rotate_index = GetRotation(code_matrix);
            if (0 <= rotate_index) {
                // ��Ŀ �ν� ����!!!

                // ��Ŀ�� �ڵ带 ������ ����� ȸ���� ������ ������ �ش�.
                RotateMatrix(code_matrix, rotate_index);
                RotateCorner(_markers[i].corner, rotate_index, _markers[i].level % 2);

                _markers[i].ID = CalcMarkerID(code_matrix);
                // TRACE ("Marker ID = %d\n", _markers[i].ID);

                FindMarkerPos3d(&_markers[i]);

                DrawMarkerInfo(&_markers[i], dst);

                // ���� ��Ŀ �ڵ�
                cvNamedWindow("Marker Image Org", CV_WINDOW_AUTOSIZE);
                cvShowImage("Marker Image Org", img_marker);
                ShowMarkerCode(cvSize(60, 60), code_matrix);
            }
        }

        cvReleaseImage(&img_marker);
    }
}

void ExtractMarkerImage(IplImage *src, IplImage *dst, sMarkerInfo &mi)
{
    assert(src->nChannels == 1);
    assert(dst->width == dst->height);

    const float ignoring_margin = 0.f;	// ���� �̹����κ��� ��Ŀ �̹����� �����ϸ鼭 ������ �׵θ��� ����

    transform_matrix = cvCreateMat(3, 3, CV_32FC1);

    if (mi.level % 2 == 0) {
        // ������ ��Ŀ�� ������ �̹��� ���� ��ǥ
        CvPoint2D32f dest_corner_cw[4] = {
            { -ignoring_margin,				-ignoring_margin},
            { -ignoring_margin,				dst->height + ignoring_margin},
            { dst->width + ignoring_margin,	dst->height + ignoring_margin},
            { dst->width + ignoring_margin,	-ignoring_margin},
        };

        // �ҽ� �̹������� ��Ŀ�� �ڳʿ� ���� ������ ��Ŀ �̹��� ���� ����� �����ϱ� ���� ��ȯ ����� ���Ѵ�.
        cvGetPerspectiveTransform(mi.corner, dest_corner_cw, transform_matrix);
    }
    else {
        CvPoint2D32f dest_corner_ccw[4] = {
            { dst->width + ignoring_margin,	-ignoring_margin},
            { dst->width + ignoring_margin,	dst->height + ignoring_margin},
            { -ignoring_margin,				dst->height + ignoring_margin},
            { -ignoring_margin,				-ignoring_margin},
        };

        // �ҽ� �̹������� ��Ŀ�� �ڳʿ� ���� ������ ��Ŀ �̹��� ���� ����� �����ϱ� ���� ��ȯ ����� ���Ѵ�.
        cvGetPerspectiveTransform(mi.corner, dest_corner_ccw, transform_matrix);
    }

    // �ҽ� �̹��� ���� ��Ŀ�� ��Ŀ �̹����� �����Ѵ�.
    cvWarpPerspective(src, dst, transform_matrix);

    

    if (mi.level % 2 == 0) {
        cvNot(dst, dst);
    }

    cvReleaseMat(&transform_matrix);
}

void ExtractCodeFromImage(IplImage *src, double code_matrix[6][6])
{
#define PIXEL_YX(img,y,x)	(unsigned char &)img->imageData[(y)*img->widthStep + (x)]

    assert(src->width == 60 && src->height == 60);

    // ��Ŀ �̹����� 6x6 ���ڷ� �ɰ� �� ������ ���� ���� �ȼ����� ��� ���Ѵ�.
    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int yi = y / 10;
            int xi = x / 10;

            assert(yi < 6 && xi < 6);
            code_matrix[yi][xi] += PIXEL_YX(src, y, x);
        }
    }

    double min_v = 255.;
    double max_v = 0.;

    // ���� ���� 0 ~ 1 ������ ������ ����ȭ �ϸ鼭 �ִ밪�� �ּҰ��� ã�´�.
    // �ϳ��� ���ڿ��� 100���� �ȼ��� �������� �� �ȼ��� �ִ� ���� 255�̱� ������
    // ���� ���� 100*255�� �������ָ� �ȴ�.
    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
            code_matrix[y][x] /= 100. * 255;

            if (min_v > code_matrix[y][x]) min_v = code_matrix[y][x];
            if (max_v < code_matrix[y][x]) max_v = code_matrix[y][x];
        }
    }

    // �ִ밪�� �ּҰ��� �߰����� ã�´�.
    double mid_v = (min_v + max_v) / 2.;

    // �߰����� �������� �������� ������ 1.�� ����� ������ 0.�� �����Ѵ�.
    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
            code_matrix[y][x] = (code_matrix[y][x] < mid_v) ? 1. : 0.;
        }
    }
}

bool CheckParity(double code_matrix[6][6])
{
    int sum = 0;

    // �׵θ��� ��� ����� �ִ��� �˻��Ѵ�.
    // ��, �� ������ �� ���� 6���̰� �𼭸��� 4���̴ϱ� 
    // ���� 24���� �Ǿ�� �Ѵ�.
    for (int i = 0; i < 6; i++) {
        sum += (int)code_matrix[0][i];
        sum += (int)code_matrix[5][i];
        sum += (int)code_matrix[i][0];
        sum += (int)code_matrix[i][5];
    }
    if (sum != 24) return false;

    sum = 0;

    // üũ���� �˻��Ѵ�.
    // �׵θ��� ������ ���� ���� ���� ¦���� �Ǿ�� �Ѵ�.
    for (int y = 1; y < 5; y++) {
        for (int x = 1; x < 5; x++) {
            sum += (int)code_matrix[y][x];
        }
    }
    return (sum % 2 == 0);
}


int GetRotation(double code_matrix[6][6])
{
    if (code_matrix[1][1] && !code_matrix[1][4] && !code_matrix[4][4] && !code_matrix[4][1]) return 0;	// ����
    else if (!code_matrix[1][1] && code_matrix[1][4] && !code_matrix[4][4] && !code_matrix[4][1]) return 1;	// �ð�������� 90�� ȸ����
    else if (!code_matrix[1][1] && !code_matrix[1][4] && code_matrix[4][4] && !code_matrix[4][1]) return 2; // �ð�������� 180�� ȸ����
    else if (!code_matrix[1][1] && !code_matrix[1][4] && !code_matrix[4][4] && code_matrix[4][1]) return 3; // �ð�������� 270�� ȸ����
    else return -1; // ������ ���� �����̴�. ����
}


void RotateMatrix(double code_matrix[6][6], int angle_idx)
{
    if (angle_idx == 0) return;

    double cb[6][6];

    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
            switch (angle_idx) {
            case 1: cb[y][x] = code_matrix[x][5 - y];		break; // �ݽð� �������� 90�� ȸ��
            case 2: cb[y][x] = code_matrix[5 - y][5 - x];	break; // �ݽð� �������� 180�� ȸ��
            case 3: cb[y][x] = code_matrix[5 - x][y];		break; // �ݽð� �������� 270�� ȸ��
            }
        }
    }
    memcpy(code_matrix, cb, sizeof(double) * 6 * 6);
}

inline void swap(CvPoint2D32f &c1, CvPoint2D32f &c2)
{
    CvPoint2D32f temp = c1;
    c1 = c2;
    c2 = temp;
}

void RotateCorner(CvPoint2D32f corner[4], int angle_idx, int dir)
{
    CvPoint2D32f c[4];

    if (dir) {
        for (int i = 0; i < 4; ++i) {
            c[i] = corner[(i + 3 + angle_idx) % 4];
        }
        swap(c[1], c[3]);
    }
    else {
        for (int i = 0; i < 4; ++i) {
            c[i] = corner[(i + 4 - angle_idx) % 4];
        }
    }
    memcpy(corner, c, sizeof(CvPoint2D32f) * 4);
}

int CalcMarkerID(double code_matrix[6][6])
{
    int id = 0;
    if (code_matrix[4][2]) id += 1;
    if (code_matrix[3][4]) id += 2;
    if (code_matrix[3][3]) id += 4;
    if (code_matrix[3][2]) id += 8;
    if (code_matrix[3][1]) id += 16;
    if (code_matrix[2][4]) id += 32;
    if (code_matrix[2][3]) id += 64;
    if (code_matrix[2][2]) id += 128;
    if (code_matrix[2][1]) id += 256;
    if (code_matrix[1][3]) id += 512;
    if (code_matrix[1][2]) id += 1024;

    return id;
}

void FindMarkerPos3d(sMarkerInfo *marker)
{

    // ȸ��(rotation)�� �̵�(translation)�� ����Ͽ� ������ ��Ʈ���� ����
    CvMat rotation = cvMat(3, 1, CV_32FC1, marker->rotation);
    CvMat translation = cvMat(3, 1, CV_32FC1, marker->translation);

    float image_xy[4][2] = {
        { marker->corner[0].x, marker->corner[0].y },
        { marker->corner[1].x, marker->corner[1].y },
        { marker->corner[2].x, marker->corner[2].y },
        { marker->corner[3].x, marker->corner[3].y },
    };

    float object_xyz[4][3] = {
        { 0.0f,				0.0f,			0.0f },
        { 0.0f,				marker->height,	0.0f },
        { marker->width,	marker->height,	0.0f },
        { marker->width,	0.0f,			0.0f },
    };

    CvMat object_points = cvMat(4, 3, CV_32FC1, &object_xyz[0][0]);
    CvMat image_points = cvMat(4, 2, CV_32FC1, &image_xy[0][0]);

    CvMat *_intrinsic_matrix = new CvMat(intrinsic_matrix);
    CvMat *_distortion_coeffs = new CvMat(distortion_coeffs);

    // 3���� �������� ��Ŀ�� ��ġ�� ������ ã�´�.
    cvFindExtrinsicCameraParams2(&object_points, &image_points,_intrinsic_matrix, _distortion_coeffs, &rotation, &translation);
}

void ShowMarkerCode(CvSize &size, double code_matrix[6][6])
{
    // �ڵ� ������κ��� ���� ��Ŀ �ڵ带 �̹����� ��ȯ�Ͽ� ǥ���Ѵ�.

    IplImage* img = cvCreateImage(size, IPL_DEPTH_8U, 1);

    cvSet(img, CV_RGB(255, 255, 255));

    double dx = img->width / 6.;
    double dy = img->height / 6.;

    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
            if (code_matrix[y][x]) {
                cvDrawRect(img, cvPoint(cvRound(x*dx), cvRound(y*dy)),
                    cvPoint(cvRound((x + 1)*dx), cvRound((y + 1)*dy)), CV_RGB(0, 0, 0), CV_FILLED);
            }
        }
    }

    //cvNamedWindow("Marker Image", CV_WINDOW_AUTOSIZE);
    //cvShowImage("Marker Image", img);

    cvReleaseImage(&img);
}

void DrawMarkerInfo(sMarkerInfo *marker, IplImage *dst)
{
    float depth = max(marker->width, marker->height);

    // 3���� �������� ��Ŀ�� x, y, z ��ǥ�� �����Ѵ�.
    float object_xyz[4][3] = {
        {0.f,			0.f,			0.f		},
        {marker->width,	0.f,			0.f		},
        {0.f,			marker->height,	0.f		},
        {0.f,			0.f,			depth	},
    };
    float image_xy[4][2] = { 0.f, };

    //std::vector<std::vector<cv::Point2f>> image_points;
    //std::vector<std::vector<cv::Point3f>> object_points;
    //cv::Mat rotationMatrix = cv::Mat::zeros(3, 3, CV_64F);
    //cv::Mat translationVector = cv::Mat::zeros(3, 1, CV_64F);

    //CvMat rotation = cvMat(3, 1, CV_32FC1, marker->rotation);
    //CvMat translation = cvMat(3, 1, CV_32FC1, marker->translation);

    CvMat image_points = cvMat(4, 2, CV_32FC1, &image_xy[0][0]);
    CvMat object_points = cvMat(4, 3, CV_32FC1, &object_xyz[0][0]);

    CvMat rotation = cvMat(3, 1, CV_32FC1, marker->rotation);
    CvMat translation = cvMat(3, 1, CV_32FC1, marker->translation);

    CvMat *_intrinsic_matrix = new CvMat(intrinsic_matrix);
    CvMat *_distortion_coeffs = new CvMat(distortion_coeffs);

    // ��Ŀ�� x, y, z ��ǥ�� �̹����� �������� �Ѵ�.
    //cv::projectPoints(object_points, rotationMatrix, translationVector, intrinsic_matrix, distortion_coeffs, image_points);
    cvProjectPoints2(&object_points, &rotation, &translation, _intrinsic_matrix, _distortion_coeffs, &image_points);


    // ��Ŀ�� ID�� ǥ���Ѵ�.
    char buff[256];
    sprintf(buff, "     ID: %d", marker->ID);
    cvPutText(dst, buff, cvPointFrom32f(marker->corner[0]), &_font, CV_RGB(255, 0, 0));
}

#endif

void cv_ARaugmentImage(IplImage* display, IplImage* img, CvPoint2D32f srcQuad[4], double scale)
{
    IplImage* cpy_img = cvCreateImage(cvSize(img->width, img->height), 8, 3);	// To hold Camera Image Mask 
    IplImage* neg_img = cvCreateImage(cvSize(img->width, img->height), 8, 3);	// To hold Marker Image Mask
    IplImage* blank;						// to assist marker pass
    IplImage temp;

    blank = cvCreateImage(cvSize(display->width, display->height), 8, 3);
    cvZero(blank);
    cvNot(blank, blank);

    CvPoint2D32f dispQuad[4];
    CvMat* disp_warp_matrix = cvCreateMat(3, 3, CV_32FC1);    // Warp matrix to store perspective data required for display

    if (scale == CV_AR_DISP_SCALE_FIT)
    {
        dispQuad[0].x = 0;				// Positions of Display image (not yet transposed)
        dispQuad[0].y = 0;

        dispQuad[1].x = display->width;
        dispQuad[1].y = 0;

        dispQuad[2].x = display->width;
        dispQuad[2].y = display->height;

        dispQuad[3].x = 0;
        dispQuad[3].y = display->height;
    }
    else
    {
        dispQuad[0].x = (display->width / 2) - (CV_AR_MARKER_SIZE / scale);			// Positions of Display image (not yet transposed)
        dispQuad[0].y = (display->height / 2) - (CV_AR_MARKER_SIZE / scale);

        dispQuad[1].x = (display->width / 2) + (CV_AR_MARKER_SIZE / scale);
        dispQuad[1].y = (display->height / 2) - (CV_AR_MARKER_SIZE / scale);

        dispQuad[2].x = (display->width / 2) + (CV_AR_MARKER_SIZE / scale);
        dispQuad[2].y = (display->height / 2) + (CV_AR_MARKER_SIZE / scale);

        dispQuad[3].x = (display->width / 2) - (CV_AR_MARKER_SIZE / scale);
        dispQuad[3].y = (display->height / 2) + (CV_AR_MARKER_SIZE / scale);
    }
  
    cvGetPerspectiveTransform(dispQuad, srcQuad, disp_warp_matrix);	// Caclculate the Warp Matrix to which Display Image has to be transformed
   

    // Note the jugglery to augment due to OpenCV's limiation passing two images [- Marker Img and Raw Img] of DIFFERENT sizes 
    // while using "cvWarpPerspective".  

    cvZero(neg_img);
    cvZero(cpy_img);
    cvWarpPerspective(display, neg_img, disp_warp_matrix);
    cvWarpPerspective(blank, cpy_img, disp_warp_matrix);
    
    cvNot(cpy_img, cpy_img);
    IplImage *mask_neg = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    cvCvtColor(neg_img, mask_neg, CV_RGB2GRAY);
    cvThreshold(mask_neg, mask_neg, 1, 255, CV_THRESH_BINARY);
    cvAnd(cpy_img, neg_img, img, mask_neg);
    cvOr(img, neg_img, img);
    
    
    // Release images
    cvReleaseImage(&cpy_img);
    cvReleaseImage(&neg_img);

    cvReleaseMat(&disp_warp_matrix);

}
