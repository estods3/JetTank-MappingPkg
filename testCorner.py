#!/usr/bin/env python2
import sys
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from visual_odometry import PinholeCamera, VisualOdometry

class SLAMrunner:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/image_raw", CompressedImage, self.callback, queue_size=1)

    def callback(self, data):
        ## Get Image
        try:
            #img = self.bridge.imgmsg_to_cv2(data, "mono8")
            np_arr = np.fromstring(data.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        except CvBridgeError as e:
            print(e)

        ## Upscale Image Size
        scale_percent = 100 * 20 * 2 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        print(width)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


        gray = np.float32(img)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[255]

        cv2.imshow('dst',img)
        cv2.waitKey(1)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

def main(args):
    slam = SLAMrunner()
    rospy.init_node('pc_map_maker_node', anonymous=True)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


