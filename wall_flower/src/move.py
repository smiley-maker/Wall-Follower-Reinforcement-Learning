#!/usr/bin/env python3

#Imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import random
import math

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

    mode = "train"
    #(y,x,w)
    startPoses = [(-1.75, 1.8, 0), (1.5, -1, 0), (1.5, 1, 1)]

    def __init__(self, mode="train"):
        """
        Constructor initializing the node, services, publisher, and subscriber. 
        """        
        Move.mode = mode
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
        if Move.mode == "train":
            self.learning()
        elif Move.mode == "test":
            self.runFile()
    
    def init_services(self):
        """
        Generates a rospy service using gazebo's reset world. This creates a 
        function to reset the world later using ServiceProxy. 
        """        
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.wait_for_service("gazebo/pause_physics")
        self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    
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
    
    def makeModelState(self, pos):
        model_msg = ModelState()
        model_msg.model_name = "turtlebot3_waffle"
        model_msg.pose.position.x = pos[0]
        model_msg.pose.position.y = pos[1]
        model_msg.pose.position.z = 0
        model_msg.pose.orientation.x = 0
        model_msg.pose.orientation.y = 0
        model_msg.pose.orientation.z = 0
        model_msg.pose.orientation.w = pos[2]
        return model_msg
    
    def splitRange(self, ranges):
        """
        This function takes a list of ranges (distance to an object) from the
        robot and sorts and gets the minimum value for three quadrants of the
        lidar sensor. 

        Args:
            ranges (List): List of range values based on 360 degrees of data. 
        """     
 #       for i,r in enumerate(ranges):
#            if r == "inf":
#                ranges[i] = 3.6
        ranges = list(ranges)
        for i, r in enumerate(ranges):
            if r > 3.5:
                ranges[i] = 3.6
        front = [r for r in ranges[339:359] if r != "inf"] + [c for c in ranges[0:19] if c != "inf"]
        front = min(front)
        left = [i for i in ranges[20:134] if i != "inf"]
        left = min(left)
        right = [j for j in ranges[224:338] if j != "inf"]
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
    
    
    def updateQValue(self, reward, Q, old_state, new_state, action, alpha=0.1, gamma=0.7, strategy="sarsa", new_action=None):
 #       temp_diff = reward + gamma*(max(Q[new_state].values())) #[new action] sarsa, get rid of max and values
#        newQ = Q[old_state][action] + alpha*(temp_diff -Q[old_state][action])

        if strategy == "sarsa" and new_action:
            temp_diff = reward + gamma*(Q[new_state][new_action])
        else:
            temp_diff = reward + gamma*(max(Q[new_state].values()))
        newQ = Q[old_state][action] + alpha*(temp_diff-Q[old_state][action])
        Q[old_state][action] = newQ
    
 
    def getStringState(self, state, cmin, mmin, fmin):
        res = ""
        if cmin <= state[0] <= mmin:
            res += "forward: close, "
        elif mmin <= state[0] <= fmin:
            res += "forward: medium, "
        elif fmin <= state[0] <= 4:
            res += "forward: far, "
        
        if cmin <= state[2] < mmin:
            res += "right: close, "
        elif mmin <= state[2] < fmin:
            res += "right: medium, "
        elif fmin <= state[2] <= 4:
            res += "right: far, "

        if cmin <= state[1] < mmin:
            res += "left: close"
        elif mmin <= state[1] < fmin:
            res += "left: medium"
        elif fmin <= state[1] <= 4:
            res += "left: far"
        
        return res

    def calculateReward(self, state, mmin, fmin):
        reward = 0
        if state[1] < mmin or state[1] >= fmin or state[0] < mmin or state[2] < mmin:
            reward = -1
        else:
            reward = 0
   #     if state[1] < mmin or state[1] >= fmin or state[0] < mmin or state[2] < mmin:
  #          reward += -1
 #       else: 
#            reward += 1
        #right, front
#        if state[2] < mmin or state[0] < mmin:
 #           reward -= 1
  #      if state[1] >= fmin:
   #         reward -= 5
    #    elif state[1] < mmin:
     #       reward -= 2
      #  else:
       #     reward += 1
        return reward

    def rewardState(self, ranges, mmin, fmin):
        state = self.splitRange(ranges)
        reward = self.calculateReward(state, mmin, fmin)
        state = self.getStringState(state, 0, mmin, fmin)
        return reward, state
    
    def plotLearning(self, episodes, data, optionalData=None):
        fig, (learning, ax) = plt.subplots(2, 1)

        numEpisodes = range(len(data))

        learning.plot(numEpisodes, data)
        if optionalData:
            ax.plot(episodes, optionalData)

        plt.savefig("learning3.pdf")
        plt.close()
        
 
    def episode(self, Q, eps):
#        self.reset_world()
        options = [self.makeModelState(p) for p in self.startPoses]
        try:
            self.set_state(np.random.choice(options))
        except:
            print("didn't work")
        

        steps = 0
        counter = 0
        total = 0
        correct = 0
        val = 0
        T = 10
        termination = False
        Move.scan = None

        while not Move.scan and not rospy.is_shutdown():
            self.publishTwist([0,0])
            rospy.sleep(0.1)
        
        reward, state = self.rewardState(Move.ranges, 0.5, 0.7)

        if random.random() < eps:
            action = np.random.choice(list(Q[state].keys()))
        else:
            action = max(Q[state], key=Q[state].get)
   
        self.publishTwist(Move.twists[action])

        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.1)
            self.pause_physics()
            steps += 1

            reward, newState = self.rewardState(Move.ranges, 0.5, 0.7 )

            if "left: far" in state:
                counter += 1
            else:
                counter = 0
            if counter >= 200:
                if total:
                    val = correct/total
                termination = True
            for d in Move.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2: 
                        reward = -15
                        if total:
                            val = correct/total
                          #  print(val)
                        termination = True
            if steps >= 800:
    #            reward += 100
                if total:
                    val = correct/total
                termination = True     

           # self.updateQValue(reward, Q, state, newState, action)

            

            if random.random() < eps:
                s = sum([math.exp(a/T) for a in Q[newState].values()])
                p = [math.exp(a/T)/s for a in Q[newState].values()]
                new_action = np.random.choice(list(Q[newState].keys()), p=p)
            else:
                new_action = max(Q[newState], key=Q[newState].get)
                if state == "forward: close, right: far, left: close" or state == "forward: medium, right: far, left: close" or state=="forward: medium, right: far, left: medium":
                    total += 1
                    if new_action == "right":
                        correct += 1
            
            self.unpause_physics()
            self.publishTwist(self.twists[new_action])
            self.updateQValue(reward, Q, state, newState, action, new_action=new_action, strategy="sarsa")
            action = new_action
            state = newState

          #  rospy.sleep(0.4)
        return val, total, reward
                    
    def learning(self):
        eps = 0.9
        duration = 800
        data = []
        dataEpisodes = 0
        for episode in range(duration):
#            self.reset_world()
            if not rospy.is_shutdown():
                print(episode)
                Q = self.getTable("CurrentQ.json")
#                Q = self.getTable("CurrentQ_" + str(episode) + ".json")
                x, total, reward = self.episode(Q, eps)
                if total > 2:
                    data.append(x)
                    dataEpisodes += 1
                    self.plotLearning(dataEpisodes, data)
 #               name = "CurrentQ_" + str(episode+1) + ".json"
                name = "CurrentQ.json"
                self.saveTable(Q, name)
            #    if episode % 20 == 0:
               #     print("Running demo....")
                #    self.runFile()
#                self.plotLearning(dataEpisodes, data)
                eps -= (0.9-0.1)/duration
    

    def runFile(self):
        steps = 0
        while not Move.scan and not rospy.is_shutdown():
            options = [self.makeModelState(p) for p in self.startPoses]
            try:
                self.set_state(np.random.choice(options))
            except:
                print("didn't work")
            rospy.sleep(0.1)
        Q = self.getTable("CurrentQ.json")
        print("table loaded")
        termination = False
        Move.scan = None
        
        reward, state = self.rewardState(Move.ranges, 0.5, 1.1)

        action = max(Q[state], key=Q[state].get)
   
        self.publishTwist(Move.twists[action])

        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.5)
            steps += 1

            reward, newState = self.rewardState(Move.ranges, 0.5, 1.1)

            for d in Move.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2:
                        termination = True
            if steps >= 800:
    #            reward += 100
                termination = True            

            state = newState
            print(state)

            action = max(Q[state], key=Q[state].get)
            print(action)
    
            self.publishTwist(self.twists[action])





    def callback(self, dist):
        """
        Callback for the subscriber. Generates decisions, thresholds, states, and
        publishes twists. 

        Args:
            dist (Dictionary): full set of data returned from the subscriber. 
        """        
        Move.scan = True
#        dist.ranges = list(np.clip(dist.ranges, 0, 3.5))
        Move.ranges = dist.ranges



if __name__ == "__main__":
    s = Move(mode="train")