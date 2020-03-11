import numpy as np
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
kMinNumFeature = 70

#TODO this may need changed
lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

## Optical Flow Feature Tracking
# Tracks corners from one image to another
def featureTracking(image_ref, image_cur, px_ref):
        print("feature Tracking: px_ref=")
        print(px_ref)
        kp1 = 0
        kp2 = 0
        if px_ref is not [] and not isinstance(px_ref, int) and px_ref.all() != 0:
	    kptwo, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
            if(st is not None):
	        st = st.reshape(st.shape[0])
	        kp1 = px_ref[st == 1]
	        kp2 = kptwo[st == 1]

	return kp1, kp2

## Camera Model
# Defines the camera model using the params specified.
class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]

## SLAM Algorithm using VisualOdometry
# Monitors change in corner positions from image to image and compute change in camera coordinates between images.
class VisualOdometry:
	def __init__(self, cam):
            self.frame_stage = 0
            self.cam = cam
            self.new_frame = None
            self.last_frame = None
            self.cur_R = None
            self.cur_t = None
            self.px_ref = None
            self.px_cur = None
            self.focal = cam.fx
            self.pp = (cam.cx, cam.cy)

        ## Find Corner Features
        # Finds corners in Images using Harris Method
        def findCorners(self, img):
            # Convert to Grayscale and locate Corners using Harris Method
            gray = np.float32(img)
            dstHar = cv2.cornerHarris(gray,2,3,0.04)

            # Annotate Image for Output
            dst = cv2.dilate(dstHar, None)
            img_annotated = img.copy()
            img_annotated[dst>0.01*dst.max()]=[255]

            # Get Corner Coordinates
            ret, dst = cv2.threshold(dstHar,0.1*dstHar.max(),255,0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

            return img_annotated, corners

        ## Process Next Frame
        # Perform feature tracking between this frame and the the last frame and detect the pose of the camera
	def processSecondFrame(self, corners):
            resetCorners = True
	    if(self.px_ref is not []):
                # Find how corners moved from last_frame to new_frame
                self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
                print("and now")
                print(self.px_ref)
                if(self.px_ref is not [] and len(self.px_ref) > 5):
		    E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    print("Process 2nd Frame: E=")
                    print(E)
		    if(E is not None):
                        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
                        print("Pose Recovered! self.cur_t=")
                        print(self.cur_t)
                        resetCorners = False
                        self.px_ref = self.px_cur

            if resetCorners:
                # No features found in last frame. use this frames features, instead, for the next iteration
                self.px_ref = corners

        ## Update
        # Updates algorithm flow with new input frame
	def update(self, img, frame_id):
            # Check New Image Frame Fits Camera Criteria
            assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
            self.new_frame = img

            # Find Corners in New Frame
            frame_annotated, corners = self.findCorners(self.new_frame)

            # Process New Frame: Either its the very First Frame, or any frame >= 2
            if(self.frame_stage == STAGE_FIRST_FRAME):
                self.px_ref = corners
                print("Process 1st Frame: Corners=")
                print(corners)
                self.frame_stage = STAGE_SECOND_FRAME
            elif(self.frame_stage == STAGE_SECOND_FRAME):
                self.processSecondFrame(corners)

            # Store Frame Before Recieving Next Frame
            self.last_frame = self.new_frame.copy()

            # Return Annotated Frame for Output
            return frame_annotated
