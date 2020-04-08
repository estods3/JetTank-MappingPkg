#!/usr/bin/env python2
import sys
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int16
from visual_odometry import PinholeCamera, VisualOdometry

class SLAMrunner:
    def __init__(self):
        #Camera Params = fx: 848.993, fy: 840.167, cx: 374.031,	cy: 222.626
        self.camModel = PinholeCamera(640, 480, 848.993, 840.167, 374.031, 222.626)
        self.vo = VisualOdometry(self.camModel)
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        self.image_sub = rospy.Subscriber("/camera/image_raw", CompressedImage, self.imageCallback, queue_size=1)
        self.state_sub = rospy.Subscriber("jt_linefollowing_motorcontrol_command", Int16, self.updateOrientation, queue_size=1)
        self.img_count = 0
        self.pose_x_mm = 0 # horizontal (lateral) width dimension
        self.pose_y_mm = 50 # vertical height dimension (height of camera)
        self.pose_z_mm = 0 # image depth (longitudinal) dimension
        self.robot_speed_mm_s = 50 # robot longitudinal speed (mm/s)

        self.x_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.z_avg = [0, 0, 0]

        self.orientationState = np.array([[1, 0],[0, 1]])

    def updateOrientation(self, data):
        ## Get pose from 'data'
        command = data.data
        if(command == 6):
            yaw = 2 # 2 degrees CCW
        elif(command == 7):
            yaw = -2 # 2 degrees CW
        else:
            yaw = 0 # yaw is straight ahead

        ## Update orientationState with respect to original position
        y = np.deg2rad(yaw)
        self.orientationState = np.array([[np.cos(y), -1*np.sin(y)], [np.sin(y), np.cos(y)]]) * self.orientationState

    def imageCallback(self, data):
        ## Get Image from 'data'
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            self.img_count = self.img_count + 1
        except CvBridgeError as e:
            print(e)

        ## Upscale Image to Original Size
        scale_percent = 100 * 20 * 2 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        ## Get Visual Odometry Coordinates
        imgAnnotated = self.vo.update(img, self.img_count)
        pose_current_xyz = self.vo.cur_t
        if(self.img_count > 2):
            x, y, z = pose_current_xyz[0], pose_current_xyz[1], pose_current_xyz[2]
	else:
            x, y, z = 0., 0., 0.

        ## Update SLAM Robot Pose
        a = np.matmul(self.orientationState, np.vstack([x,z]))
        print(a)
        self.x_avg.insert(0, a[0])
        self.x_avg.pop()
        self.z_avg.insert(0, a[1])
        self.z_avg.pop()
        self.pose_x_mm = self.pose_x_mm + -1*(self.robot_speed_mm_s * 1.0/30 * sum(self.x_avg)/len(self.x_avg)) # x-component motion
        self.pose_z_mm = self.pose_z_mm + -1*(self.robot_speed_mm_s * 1.0/30 * sum(self.z_avg)/len(self.z_avg)) # z-component motion

        ## Draw Results on Map
        origin = [490, 490]
        draw_x, draw_z = int(self.pose_x_mm) + origin[0], int(self.pose_z_mm) + origin[1]
        cv2.circle(self.traj, (draw_x,draw_z), 1, (0,0,255), 1)
        cv2.rectangle(self.traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fmm y=%2fmm z=%2fmm" % (self.pose_x_mm, self.pose_y_mm, self.pose_z_mm)
        cv2.putText(self.traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        ## Show Image and Map
        cv2.imshow('Road facing camera', imgAnnotated)
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
