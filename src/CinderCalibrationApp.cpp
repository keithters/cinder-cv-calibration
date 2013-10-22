#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"

#include "CinderOpenCV.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#define BOARD_IMG_COUNT		10		// number of images to be used for camera callibration
#define BOARD_CORNERS_X		8		// number of square intersections on the chessboard along x axis
#define BOARD_CORNERS_Y		6		// number of square intersections on the chessboard along y axis

#define CAPTURE_WIDTH		640		// pixel width of camera input
#define CAPTURE_HEIGHT		480		// pixel height of camera input

#define STATE_DETECT		0
#define STATE_TAKE_IMG		1
#define STATE_CALIBRATING	2
#define STATE_CALIBRATED	3

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
	
	CaptureRef				mCapture;
	Surface					mCaptureSurf;
	cinder::gl::TextureRef	mCaptureTex;
	
	Mat mCaptureMat;
	Mat mGrayMat;
	Mat mUndistortedMat;
	Mat intrinsic;
	Mat distCoeffs;
	
	vector<Point3f> obj;
	
	vector<vector<Point3f> > mObjectPoints;
    vector<vector<Point2f> > mImagePoints;
	
	int mState;
	int mImages;

};

void CinderCalibrationApp::prepareSettings( cinder::app::AppBasic::Settings *settings ) {
    settings->setFrameRate( 60.0 );
    settings->setWindowSize( CAPTURE_WIDTH, CAPTURE_HEIGHT );
}

void CinderCalibrationApp::setup()
{
	mState = STATE_DETECT;
	mImages = 0;
	
	int numSquares = BOARD_CORNERS_X * BOARD_CORNERS_Y;
	for( int j = 0;j < numSquares; j++ ) {
		obj.push_back( Point3f( j / BOARD_CORNERS_X, j % BOARD_CORNERS_Y, 0.0f ) );
	}

	try {
		mCapture = Capture::create( CAPTURE_WIDTH, CAPTURE_HEIGHT );
		mCapture->start();
	} catch( ... ) {
		console() << "Failed to initialize capture" << std::endl;
	}
}

void CinderCalibrationApp::keyDown( KeyEvent event )
{
	if ( event.getChar() == ' ' && mState == STATE_DETECT ) {
		mState = STATE_TAKE_IMG;
	} else if ( event.getChar() == 's' ) {
		// save the camera params yaml file
	}
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
		
		if ( findChessboardCorners( mCaptureMat, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS ) ) {
			cv::cornerSubPix( mGrayMat, corners, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ) );
			cv::drawChessboardCorners( mGrayMat, boardSize, corners, true );
		}
		
		// capture image and object points for callibration
		if ( mState == STATE_TAKE_IMG ) {
			
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
			intrinsic.ptr<float>(0)[0] = 1;
			intrinsic.ptr<float>(1)[1] = 1;
			double rms = calibrateCamera( mObjectPoints, mImagePoints, mCaptureMat.size(), intrinsic, distCoeffs, rvecs, tvecs );
			
			console() << "Re-projection error reported by calibrateCamera: " << rms << endl;
			
			mState = STATE_CALIBRATED;
			
//			// Check for projection errors
//			vector<Point2f> imagePoints2;
//			int i, totalPoints = 0;
//			double totalErr = 0, err;
//			vector<float> perViewErrors;
//			perViewErrors.resize( mObjectPoints.size() );
//
//			for( i = 0; i < (int)mObjectPoints.size(); ++i ) {
//			
//				projectPoints( Mat( mObjectPoints[i] ), rvecs[i], tvecs[i], intrinsic, distCoeffs, imagePoints2 );
//				err = norm( Mat( mImagePoints[i] ), Mat( imagePoints2 ), CV_L2 );
//
//				int n = (int)mObjectPoints[i].size();
//				perViewErrors[i] = (float) std::sqrt( err * err /n );
//				totalErr        += err * err;
//				totalPoints     += n;
//			}
//
//			console() << "Calibration Error = " << sqrt( totalErr / totalPoints ) << "%" << endl;;
		}
		
		if ( mState == STATE_CALIBRATED ) {
			undistort( mCaptureMat, mUndistortedMat, intrinsic, distCoeffs );
		}
		
		// save camera params to YAML file for future use
	}
}

void CinderCalibrationApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
	
	if ( mCaptureTex ){
	
		if ( mState == STATE_CALIBRATED ) {
			gl::draw( fromOcv( mUndistortedMat ) );
		} else {
			gl::draw( fromOcv( mGrayMat ) );
		}
	}
}

CINDER_APP_NATIVE( CinderCalibrationApp, RendererGl )
