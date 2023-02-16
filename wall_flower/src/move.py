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
        "right": [0.2, 0.95],
        "left": [0.2, -0.95],
        "forward": [0.25, 0]
    }
    scan = None
    ranges = None

    def __init__(self):
        """
        Constructor initializing the node, services, publisher, and subscriber. 
        """        
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
#        print("hi")
        self.learning()
    
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
    
    def saveTable(self, table, filename):
        with open(filename, "w") as f:
            json.dump(table, f, indent=4)
    
    def calculateResult(self, ranges):
        state = self.splitRange(ranges)
        #reward calculation
        
        res = ""
        close_min = 0
        medium_min = 0.7
        far_min = 1.7
        far_max = 4
        freward = 0
        lreward = 0
        rreward = 0

        #forward
        if close_min <= state[0] < medium_min:
            res += "forward: close, "
            freward = -5
        if medium_min <= state[0] < far_min:
            res += "forward: medium, "
            freward = 0
        if far_min <= state[0] <= far_max:
            res += "forward: far, "
            freward = 0
    
        #right
        if close_min <= state[1] < medium_min:
            res += "right: close, "
            rreward = -5
        if medium_min <= state[1] < far_min:
            res += "right: medium, "
            rreward = 10
        if far_min <= state[1] <= far_max:
            res += "right: far, "
            rreward = -5
        
        #left
        if close_min <= state[2] < medium_min:
            res += "left: close"
            lreward = -5
        if medium_min <= state[2] < far_min:
            res += "left: medium"
            lreward = -1
        if far_min <= state[2] <= far_max:
            res += "left: far"
            lreward = -5
        
        reward = lreward + rreward + freward
        
        return res, reward
    
    def updateQValue(self, reward, Q, old_state, new_state, action, alpha=0.2, gamma=0.8):
        temp_diff = reward + gamma*(max(Q[new_state].values()))
        newQ = Q[old_state][action] + alpha*(temp_diff -Q[old_state][action])
        Q[old_state][action] = newQ
#        return newQ
    
 
    def getStringState(self, state, cmin, mmin, fmin):
        res = ""
        if cmin <= state[0] < mmin:
            res += "forward: close, "
        elif mmin <= state[0] < fmin:
            res += "forward: medium, "
        elif fmin <= state[0] <= 4:
            res += "forward: far, "
        
        if cmin <= state[1] < mmin:
            res += "right: close, "
        elif mmin <= state[1] < fmin:
            res += "right: medium, "
        elif fmin <= state[1] <= 4:
            res += "right: far, "

        if cmin <= state[2] < mmin:
            res += "left: close"
        elif mmin <= state[2] < fmin:
            res += "left: medium"
        elif fmin <= state[2] <= 4:
            res += "left: far"
        
        return res


 
    def episode(self, Q):
        self.reset_world()
        Move.scan = None
        while not Move.scan and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        state, reward = self.calculateResult(Move.ranges)
#        res = self.getStringState(state, 0, 0.7, 2.1)
        action = max(Q[state], key=Q[state].get)
        self.publishTwist(Move.twists[action])
        termination = False
#        print(Move.scan)
        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.1)
            newState, reward = self.calculateResult(Move.ranges)
            self.updateQValue(reward, Q, state, newState, action)
            state = newState
            action = max(Q[state], key=Q[state].get)
            self.publishTwist(self.twists[action])
            for d in Move.ranges:
                if d < 0.2:
                    termination = True  
            
    def learning(self):
        for episode in range(300):
            self.reset_world()
            if not rospy.is_shutdown():
                print(episode)
                Q = self.getTable("CurrentQ.json")
#                Q = self.getTable("CurrentQ_" + str(episode) + ".json")
                self.episode(Q)
 #               name = "CurrentQ_" + str(episode+1) + ".json"
                name = "CurrentQ.json"
                self.saveTable(Q, name)


    def callback(self, dist):
        """
        Callback for the subscriber. Generates decisions, thresholds, states, and
        publishes twists. 

        Args:
            dist (Dictionary): full set of data returned from the subscriber. 
        """        
        Move.scan = True
        dist.ranges = list(np.clip(dist.ranges, 0, 3.5))
        Move.ranges = dist.ranges
#        self.learning(dist.ranges)
        #Resets the robot if it is really close to a wall. 
        # for d in dist.ranges: 
        #     if d < 0.2: 
        #         self.reset_world() #resets the world

        # state = self.splitRange(dist.ranges) #Gets the state in terms of distance from the front, left, and right sides
        
        # #This section formats the input data as a string to compare it with the Q-Table. 
        # res = ""
        # close_min = 0
        # close_max = 0.7
        # medium_min = 0.7
        # medium_max = 1.7
        # far_min = 1.7
        # far_max = 3.5
        # #forward
        # if close_min <= state[0] <= close_max:
        #     res += "forward: close, "
        # if medium_min <= state[0] <= medium_max:
        #     res += "forward: medium, "
        # if far_min <= state[0] <= far_max:
        #     res += "forward: far, "

        # #right
        # if close_min <= state[1] <= close_max:
        #     res += "right: close, "
        # if medium_min <= state[1] <= medium_max:
        #     res += "right: medium, "
        # if far_min <= state[1] <= far_max:
        #     res += "right: far, "
        
        # #left
        # if close_min <= state[2] <= close_max:
        #     res += "left: close"
        # if medium_min <= state[2] <= medium_max:
        #     res += "left: medium"
        # if far_min <= state[2] <= far_max:
        #     res += "left: far"
        
        # q = self.getTable("Q.json") #Gets the q table
        # print(q)
        # #Calculates the maximum utility action as defined in the dictionary.
        # val = max(q[res], key=q[res].get)
        # #Publishes the twists to send the robot a move command. 
        # self.publishTwist(self.twists[val]) 


if __name__ == "__main__":
    s = Move()