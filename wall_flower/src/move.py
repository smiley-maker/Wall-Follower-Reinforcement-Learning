#!/usr/bin/env python3

#Imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import json
import numpy as np

class Move():
    #Defines the possible action set and how they will be represented using twists. 
    #The first entry gives the velocity in the x direction, and the second is the 
    #angular velocity around z. 
    twists = {
        "right": [0.25, 0.95],
        "left": [0.25, -0.95],
        "forward": [0.3, 0]
    }

    def __init__(self):
        """
        Constructor initializing the node, services, publisher, and subscriber. 
        """        
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
    
    def init_services(self):
        """
        Generates a rospy service using gazebo's reset world. This creates a 
        function to reset the world later using ServiceProxy. 
        """        
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    
    def init_node(self):
        #Initializes a node for this project
        rospy.init_node("wallFlower")
    
    def init_publisher(self):
        #Initializes a publisher to publish Twist messages on cmd_vel to move the robot. 
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    def init_subscriber(self):
        #Sets up a subscriber from rospy that consumers the Laserscan /scan topic. 
        sub = rospy.Subscriber("/scan", LaserScan, self.callback)
        rospy.spin()
    
    def publishTwist(self, twister):
        """
        Helper method to publish twists using the publisher defined above. 

        Args:
            twister (List): [linear velocity in x, angular velocity in z]
        """        
        t = Twist()
        t.linear.x = twister[0]
        t.linear.y = 0
        t.linear.z = 0
        t.angular.x = 0
        t.angular.y = 0
        t.angular.z = twister[1]
        self.pub.publish(t)

    
    def splitRange(self, ranges):
        """
        This function takes a list of ranges (distance to an object) from the
        robot and sorts and gets the minimum value for three quadrants of the
        lidar sensor. 

        Args:
            ranges (List): List of range values based on 360 degrees of data. 
        """        
        front = [r for r in ranges[339:359] if r != "inf"] + [c for c in ranges[0:19] if c != "inf"]
        front = min(front)
        left = [i for i in ranges[44:134] if i != "inf"]
        left = min(left)
        right = [j for j in ranges[224:314] if j != "inf"]
        right = min(right)
        return (front, left, right)
    
    def getTable(self, filename):
        """
        Opens the Q table that I defined (initially created in the make_table file)

        Returns:
            dictionary: Queue table in dictionary format
        """        
        with open(filename, "r") as f:
            q = json.load(f)
        return q
    

    def callback(self, dist):
        """
        Callback for the subscriber. Generates decisions, thresholds, states, and
        publishes twists. 

        Args:
            dist (Dictionary): full set of data returned from the subscriber. 
        """        
        dist.ranges = list(np.clip(dist.ranges, 0, 3.5))
        #Resets the robot if it is really close to a wall. 0.12 meters is the minimum sensor
        #range, so I thought it would be a reasonable value here. 
        for d in dist.ranges: 
            if d < 0.12: 
                self.reset_world() #resets the world

        state = self.splitRange(dist.ranges) #Gets the state in terms of distance from the front, left, and right sides
        
        #This section formats the input data as a string to compare it with the Q-Table. 
        res = ""
        close_min = 0
        close_max = 0.7
        medium_min = 0.7
        medium_max = 1.7
        far_min = 1.7
        far_max = 3.5
        #forward
        if close_min <= state[0] <= close_max:
            res += "forward: close, "
        if medium_min <= state[0] <= medium_max:
            res += "forward: medium, "
        if far_min <= state[0] <= far_max:
            res += "forward: far, "

        #right
        if close_min <= state[1] <= close_max:
            res += "right: close, "
        if medium_min <= state[1] <= medium_max:
            res += "right: medium, "
        if far_min <= state[1] <= far_max:
            res += "right: far, "
        
        #left
        if close_min <= state[2] <= close_max:
            res += "left: close"
        if medium_min <= state[2] <= medium_max:
            res += "left: medium"
        if far_min <= state[2] <= far_max:
            res += "left: far"
        
        q = self.getTable("Q.json") #Gets the q table
        #Calculates the maximum utility action as defined in the dictionary.
        val = max(q[res], key=q[res].get)
        #Publishes the twists to send the robot a move command. 
        self.publishTwist(self.twists[val])

if __name__ == "__main__":
    s = Move()