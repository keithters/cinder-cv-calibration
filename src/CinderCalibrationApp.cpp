#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
#include "cinder/Xml.h"

#include "CinderOpenCV.h"

#include <fstream>

#define BOARD_CORNERS_X		8		// number of square intersections on the chessboard along x axis
#define BOARD_CORNERS_Y		6		// number of square intersections on the chessboard along y axis

#define BOARD_IMG_COUNT		10		// number of images to be used for camera callibration

#define CAPTURE_WIDTH		640		// pixel width of camera input
#define CAPTURE_HEIGHT		480		// pixel height of camera input

#define STATE_DETECT		0
#define STATE_TAKE_IMG		1
#define STATE_CALIBRATING	2
#define STATE_CALIBRATED	3

#define OUTPUT_FILE	"Users/keith/Desktop/params.yaml"

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace cv;

class CinderCalibrationApp : public AppNative {
  public:
	void prepareSettings( cinder::app::AppBasic::Settings *settings );
	void setup();
	void keyDown( KeyEvent event );
	void update();
	void draw();
	bool saveCameraParams();
	
	CaptureRef				mCapture;
	Surface					mCaptureSurf;
	cinder::gl::TextureRef	mCaptureTex;
	
	Mat mCaptureMat;
	Mat mGrayMat;
	Mat mUndistortedMat;
	Mat intrinsic;
	Mat distortion;
	
	vector<Point3f> obj;
	
	vector<vector<Point3f> > mObjectPoints;
    vector<vector<Point2f> > mImagePoints;
	
	int mPattern;
	int mState;
	int mImages;
	
	bool showDistorted;
	double avgError;

};

void CinderCalibrationApp::prepareSettings( cinder::app::AppBasic::Settings *settings ) {
    settings->setFrameRate( 60.0 );
    settings->setWindowSize( CAPTURE_WIDTH, CAPTURE_HEIGHT );
}

void CinderCalibrationApp::setup()
{
	mState			= STATE_DETECT;
	mImages			= 0;
	showDistorted	=	true;

	try {
		mCapture = Capture::create( CAPTURE_WIDTH, CAPTURE_HEIGHT );
		mCapture->start();
		console() << mCapture->getSize() << endl;;
		console() << getWindowSize() << endl;
	} catch( ... ) {
		console() << "Failed to initialize capture" << std::endl;
	}
	
	int numSquares = BOARD_CORNERS_X * BOARD_CORNERS_Y;
	for( int j = 0;j < numSquares; j++ ) {
		obj.push_back( Point3f( j / BOARD_CORNERS_X, j % BOARD_CORNERS_X, 0.0f ) );
	}
}

void CinderCalibrationApp::keyDown( KeyEvent event )
{
	if ( event.getChar() == ' ' && mState == STATE_DETECT ) {
		mState = STATE_TAKE_IMG;
	}
	
	if ( event.getChar() == 's' && mState == STATE_CALIBRATED ) {
		saveCameraParams();
	}
	
	if ( event.getChar() == 'd') {
		showDistorted = !showDistorted;
	}
}

bool CinderCalibrationApp::saveCameraParams()
{
	ofstream oStream( "/Users/keith/Desktop/params.yaml" );
	
	oStream << "%YAML:1.0\n";
	oStream << "image_width:" << CAPTURE_WIDTH << "\n";
	oStream << "image_height:" << CAPTURE_HEIGHT << "\n";
	oStream << "board_width:" << BOARD_CORNERS_X << "\n";
	oStream << "board_height:" << BOARD_CORNERS_Y << "\n";
	oStream << "camera_matrix: !!opencv-matrix\n";
	oStream << "   rows: 3\n   cols: 3\n   dt: d\n";
	oStream << "   data:" << intrinsic << "\n";
	oStream << "distortion_coefficients: !!opencv-matrix\n";
	oStream << "   rows: 1\n   cols: 5\n   dt: d\n";
	oStream << "   data:" << distortion << "\n";
	oStream << "avg_reprojection_error: " << avgError << "\n";
	
	oStream.close();
	return true;
}



void CinderCalibrationApp::update()
{
	if ( mCapture->checkNewFrame() ) {
		
		// Find the chessboard in the video feed
		mCaptureSurf	= mCapture->getSurface();
		mCaptureTex		= gl::Texture::create( mCaptureSurf );
		mCaptureMat		= toOcv( mCaptureSurf );
		
		cv::Size boardSize = cv::Size( BOARD_CORNERS_X, BOARD_CORNERS_Y );
		cvtColor( mCaptureMat, mGrayMat, CV_BGR2GRAY );
		vector<Point2f> corners;
		
		bool found = findChessboardCorners( mCaptureMat, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );
		
		if ( found ) {
			cornerSubPix( mGrayMat, corners, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ) );
			drawChessboardCorners( mGrayMat, boardSize, Mat(corners), true );
		}
		
		// capture image and object points for callibration
		if ( found && mState == STATE_TAKE_IMG ) {
			
			mImagePoints.push_back( corners );
            mObjectPoints.push_back( obj );
			mImages++;
			console() << "image " << mImages << " of " << BOARD_IMG_COUNT << endl;
			
			if ( mImages >= BOARD_IMG_COUNT ) {
				mState = STATE_CALIBRATING;
			} else {
				mState = STATE_DETECT;
			}
		}
		
		// calibrate camera and check for projection errors
		if ( mState == STATE_CALIBRATING ) {
			
			// Calibrate
			intrinsic = Mat(3, 3, CV_32FC1);
			vector<Mat> rvecs;
			vector<Mat> tvecs;
			intrinsic.ptr<float>(0)[0] = CAPTURE_WIDTH / CAPTURE_HEIGHT;
			intrinsic.ptr<float>(1)[1] = 1;
			avgError = calibrateCamera( mObjectPoints, mImagePoints, mCaptureMat.size(), intrinsic, distortion, rvecs, tvecs );
			
			console() << "re-projection error from calibrateCamera(): " << avgError << endl;
			
			if ( !checkRange( intrinsic ) || !checkRange( distortion ) ) {
				console() << "Calibration failed. Try again." << endl;
				exit(1);
 			}
			// void calibrationMatrixValues(InputArray cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio)
			double apertureWidth;
			double apertureHeight;
			double fovx;
			double fovy;
			double focalLength;
			Point2d principalPoint;
			double aspectRatio;
			
			calibrationMatrixValues( intrinsic, boardSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio );
			
			mState = STATE_CALIBRATED;
		}
		
		if ( mState == STATE_CALIBRATED ) {
			undistort( mCaptureMat, mUndistortedMat, intrinsic, distortion );
		}
	}
}

void CinderCalibrationApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
	
	if ( mCaptureTex ){
		if ( mState == STATE_CALIBRATED && showDistorted ) {
			gl::draw( fromOcv( mUndistortedMat ) );
		} else {
			gl::draw( fromOcv( mGrayMat ) );
		}
	}
}

CINDER_APP_NATIVE( CinderCalibrationApp, RendererGl )
