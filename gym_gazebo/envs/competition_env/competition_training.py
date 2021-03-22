
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class GazeboCompEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/andrew/enph353_gym-gazebo/gym_gazebo/envs/ros_ws/src/competition_ros_pkg/enph353/enph353_utils/launch/sim.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        frame_bw = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        bottom = frame_bw[-100:-1,:]
        h,w,_ = cv_image.shape

        # turning image binary
        ret, thresh = cv2.threshold(bottom,200,255,cv2.THRESH_BINARY_INV)
        moments = cv2.moments(thresh)

        # finding COM
        x = -1
        pixels = 0
        offline = 500
        pixels = np.count_nonzero(thresh==255)
        print(pixels)

        #enough pixels to count that the line is on screen
        if(pixels > offline and pixels < 30000):
            x = int(moments["m10"]/moments["m00"])

            i = 1
            inc = w//10
            while i*inc < w:
                if x in range((i-1)*inc,i*inc):
                    state[i-1] = 1
                    break

                i+=1
        else:
            self.timeout += 1
            print("where is the line")

        if self.timeout >= 30:
            done = True

        print(state)
        cv2.imshow("test",thresh)
        # cv2.imshow("test2",bottom)
        cv2.waitKey(3) 

        # print(state)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/R1/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 8
            elif action == 1:  # LEFT
                reward = 1
            else:
                reward = 1  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/R1/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
