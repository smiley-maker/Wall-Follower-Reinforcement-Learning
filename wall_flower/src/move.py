#!/usr/bin/env python3

#Imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import pandas as pd
from std_srvs.srv import Empty
import json
import numpy as np

class Move():
    twists = {
        "right": [0.25, 0.7],
        "left": [0.25, -0.7],
        "forward": [0.35, 0]
    }
    def __init__(self):
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
    
    def init_services(self):
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    
    def init_node(self):
        rospy.init_node("wallFlower")
    
    def init_publisher(self):
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    def init_subscriber(self):
        sub = rospy.Subscriber("/scan", LaserScan, self.callback)
        rospy.spin()
    
    def publishTwist(self, twister):
        t = Twist()
        t.linear.x = twister[0]
        t.linear.y = 0
        t.linear.z = 0
        t.angular.x = 0
        t.angular.y = 0
        t.angular.z = twister[1]
        self.pub.publish(t)

    
    def splitRange(self, ranges):
        front = [r for r in ranges[344:359] if r != "inf"] + [c for c in ranges[0:14] if c != "inf"]
        front = min(front)
        left = [i for i in ranges[44:134] if i != "inf"]
        left = min(left)
        right = [j for j in ranges[224:314] if j != "inf"]
        right = min(right)
   #     print(str(front) + "\t" + str(left) + "\t" + str(right))
        return (front, left, right)
    
    def getTable(self):
        with open("Q.json", "r") as f:
            q = json.load(f)
        return q
    
    def getTwist(self, q):
        newQ = {}
        for key in q.keys():

            if q[key]["right"] == 1:
                t = Twist()
                t.linear.x = 1
                t.linear.y = 0
                t.linear.z = 0
                t.angular.x = 0
                t.angular.x = 0
                t.angular.z = 1
                newQ[key] = t
            elif q[key]["left"] == 1:
                t = Twist()
                t.linear.x = 1
                t.linear.y = 0
                t.linear.z = 0
                t.angular.x = 0
                t.angular.x = 0
                t.angular.z = -1
                newQ[key] = t

#                print(newQ)
        return newQ        

    def callback(self, dist):
        dist.ranges = list(np.clip(dist.ranges, 0, 3.5))
        for d in dist.ranges: 
            if d < 0.2:
                self.reset_world()
    #    print("callback")
 #       rospy.loginfo(len(dist.ranges))
   #     self.reset_world()
        state = self.splitRange(dist.ranges)
        
        res = ""
        #forward
        if 0 <= state[0] <= 0.8:
            res += "forward: close, "
        if 0.8 <= state[0] <= 2.0:
            res += "forward: medium, "
        if 2.0 <= state[0] <= 3.5:
            res += "forward: far, "

        #right
        if 0 <= state[1] <= 0.8:
            res += "right: close, "
        if 0.8 <= state[1] <= 2.0:
            res += "right: medium, "
        if 2.0 <= state[1] <= 3.5:
            res += "right: far, "
        
        #left
        if 0 <= state[2] <= 0.8:
            res += "left: close"
        if 0.8 <= state[2] <= 2.0:
            res += "left: medium"
        if 2.0 <= state[2] <= 3.5:
            res += "left: far"
        
        q = self.getTable()
        val = max(q[res], key=q[res].get)
        print(val)
        self.publishTwist(self.twists[val])

if __name__ == "__main__":
    s = Move()