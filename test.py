#!/usr/bin/env python2
import sys
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage


from visual_odometry import PinholeCamera, VisualOdometry

class SLAMrunner:
    def __init__(self):
        #fx: 848.993	 fy: 840.167 	cx: 374.031 	cy: 222.626
        self.camModel = PinholeCamera(640, 480, 848.993, 840.167, 374.031, 222.626)
        self.vo = VisualOdometry(self.camModel)
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        self.image_sub = rospy.Subscriber("/camera/image_raw", CompressedImage, self.callback, queue_size=1)
        self.img_id = 0

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

	self.vo.update(img, self.img_id)

	cur_t = self.vo.cur_t

	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+290, int(z)+90
	true_x, true_y = int(self.vo.trueX)+290, int(self.vo.trueZ)+90

	cv2.circle(self.traj, (draw_x,draw_y), 1, (self.img_id*255/4540,255-self.img_id*255/4540,0), 1)
	cv2.circle(self.traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.rectangle(self.traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	cv2.putText(self.traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', self.traj)
	cv2.waitKey(1)

    def saveMap(self):
        cv2.imwrite('map.png', self.traj)

def main(args):
    slam = SLAMrunner()
    rospy.init_node('pc_map_maker_node', anonymous=True)
    rospy.spin()
    print("Map Maker: Node Shutting down")
    slam.saveMap()
    print("Map Maker: Map Saved to File")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


