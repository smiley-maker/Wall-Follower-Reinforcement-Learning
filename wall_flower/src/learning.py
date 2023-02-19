#!/usr/bin/env python3

#imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import numpy as np
import pandas as pd
import random
import json
import warnings
warnings.filterwarnings("ignore")


class Learning():
    twists = {
        "turn right": [0.2, 0.95],
        "turn left": [0.2, -0.95],
        "forward": [0.25, 0]
    }

    scan = None
    ranges = None
    df = None
    mode = "train"

    def __init__(self, mode="train"):
        Learning.mode = mode
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
        if Learning.mode == "train":
            self.learn()
        elif Learning.mode == "display":
            self.runFile()
    
    def init_services(self):
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    
    def init_node(self):
        rospy.init_node("wallFlower")
    
    def init_publisher(self):
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    def init_subscriber(self):
        self.sub = rospy.Subscriber("/scan", LaserScan, self.callback)
    
    def publishTwist(self, twister):
        t = Twist()
        t.linear.x = twister[0]
        t.linear.y = 0
        t.linear.z = 0
        t.angular.x = 0
        t.angular.y = 0
        t.angular.z = twister[1]
        self.pub.publish(t)
    
    def makeQTable(self):
        a = {
            "front right": [],
            "front left": [],
            "left": [],
            "back left": [],
            "forward": [],
            "turn left": [],
            "turn right": []
        }

        df = pd.DataFrame(a)

        base_states = ["close", "medium", "far"]

        for one in base_states:
            for two in base_states:
                for three in base_states:
                    for four in base_states:
                        df.loc[len(df.index)] = [one, two, three, four, 0, 0, 0]
        

        return df

    def splitRange(self, ranges):
        ranges = list(ranges)
        for i, r in enumerate(ranges):
            if r > 3.5:
                Learning.ranges[i] = 3.6
            
            front_right = [r for r in Learning.ranges[315:359]]
            front_right = min(front_right)
            front_left = [r for r in Learning.ranges[0:44]]
            front_left = min(front_left)
            left = [r for r in Learning.ranges[45:89]]
            left = min(left)
            back_left = [r for r in Learning.ranges[90:134]]
            back_left = min(back_left)

        return (front_right, front_left, left, back_left)
    
    def updateTable(self, df, state, actions):
        #actions - (forward #, left #, right #)
        #state - desired state in (front right, front left, left, back left) format
        toChange = df[(df["front right"] == state[0]) & (df["front left"] == state[1]) & (df["left"] == state[2]) & (df["back left"] == state[3])].index
        df["forward"][toChange] = actions[0]
        df["turn right"][toChange] = actions[1]
        df["turn left"][toChange] = actions[2]

        return df
    
    def dataframeToFile(self, df):
        df.to_json("currentData.json")


    def updateQValue(self, reward, df, old_state, new_state, action, alpha=0.2, gamma=0.8):
        #print(new_state)
        current = df[(df["front right"] == old_state[0]) & (df["front left"] == old_state[1]) & (df["left"] == old_state[2]) & (df["back left"] == old_state[3])]
        updated = df[(df["front right"] == new_state[0]) & (df["front left"] == new_state[1]) & (df["left"] == new_state[2]) & (df["back left"] == new_state[3])]
        actions = current[["forward", "turn left", "turn right"]]

        m = float(updated[max(list(updated[["forward", "turn left", "turn right"]]))])
        #print(m)

        newQ = actions[action] + alpha*(reward + gamma*(m - actions[action]))
    #    print("new Q: " + str(float(newQ)))
        df[action][current.index] = float(newQ)
    
    def getState(self, state, cmin, mmin, fmin):
        #state - forward, left, right
        strState = ""
#        print(state)
        if cmin <= state[0] < mmin:
            strState += "close "
        elif mmin <= state[0] < fmin:
            strState += "medium "
        elif fmin <= state[0] <= 4:
            strState += "far "
        

        if cmin <= state[1] < mmin:
            strState += "close "
        elif mmin <= state[1] < fmin:
            strState += "medium "
        elif fmin <= state[1] <= 4:
            strState += "far "
        
        if cmin <= state[2] < mmin:
            strState += "close "
        elif mmin <= state[2] < fmin:
            strState += "medium "
        elif fmin <= state[2] <= 4:
            strState += "far "
        
        if cmin <= state[3] < mmin:
            strState += "close"
        elif mmin <= state[3] < fmin:
            strState += "medium"
        elif fmin <= state[3] <= 4:
            strState += "far"


        strState = strState.split(" ")
      #  print(strState)
        return strState
        
    def calculateReward(self, state, mmin, fmin):
        #state: [front right, front left, left, back left]
        if min([state[1], state[2], state[3]]) < mmin or min([state[1], state[2], state[3]]) >= fmin or min([state[0], state[1]]) < mmin:
            reward = -5
        else:
            reward = 0
        return reward
    
    def rewardState(self, ranges, mmin, fmin):
        state = self.splitRange(ranges)
        reward = self.calculateReward(state, mmin, fmin)
        state = self.getState(state, 0, mmin, fmin)
        return reward, state
    
    def episode(self, eps):
        self.reset_world()

        steps = 0
        counter = 0
        randoms = 0
        termination = False
        Learning.scan = None

        while not Learning.scan and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        reward, state = self.rewardState(Learning.ranges, 0.5, 1.1)

        x = self.df[(self.df["front right"] == state[0]) & (self.df["front left"] == state[1]) & (self.df["left"] == state[2]) & (self.df["back left"] == state[3])].index
        if random.random() < eps:
            action = np.random.choice(list(self.df[["forward", "turn right", "turn left"]].iloc[x]))
        else:
            action = max(self.df[["forward", "turn right", "turn left"]].loc[x])
        self.publishTwist(Learning.twists[action])

        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.1)
            steps += 1

            reward, newState = self.rewardState(Learning.ranges, 0.5, 1.1)
            if state[1] == "far":
                counter += 1
            else:
                counter = 0
            if counter >= 150:
                termination = True
            for d in Learning.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2:
                        reward -= 75
                        termination = True
            if steps >= 800:
                termination = True
            try:
                self.updateQValue(reward, self.df, state, newState, action)
            except:
                print(newState)
                print("____________________________________")
                print(state)
                raise
            state = newState

            x = self.df[(self.df["front right"] == state[0]) & (self.df["front left"] == state[1]) & (self.df["left"] == state[2]) & (self.df["back left"] == state[3])].index
            if random.random() < eps:
                action = np.random.choice(list(self.df[["forward", "turn right", "turn left"]].iloc[x]))
            else:
                action = max(self.df[["forward", "turn right", "turn left"]].iloc[x])
            self.publishTwist(Learning.twists[action])

    def learn(self):
        eps = 0.9
        duration = 300
        self.df = self.makeQTable()
        for episode in range(duration):
            if not rospy.is_shutdown(): 
                print("Episode: " + str(episode))
                self.episode(eps)
                eps -= 1.2*(0.9-0.1)/duration
                self.dataframeToFile(self.df)
#                print(self.df)
                #self.dataframeToFile(df)
    
    def runFile(self):
        while not Learning.scan and not rospy.is_shutdown():
            self.reset_world()
            rospy.sleep(0.1)
        df = pd.read_json("currentData.json")
        print("file read")
      #  reward, state = self.rewardState(self.ranges, 0.5, 1.1)
      #  x = df[(df["front right"] == state[0]) & (df["front left"] == state[1]) & (df["left"] == state[2]) & (df["back left"] == state[3])].index
      #  action = max(df[["forward", "turn right", "turn left"]].iloc[x])
      #  self.publishTwist(Learning.twists[action])


        steps = 0
        counter = 0
        randoms = 0
        termination = False
        Learning.scan = None

        reward, state = self.rewardState(Learning.ranges, 0.5, 1.1)

        x = df[(df["front right"] == state[0]) & (df["front left"] == state[1]) & (df["left"] == state[2]) & (df["back left"] == state[3])].index
        action = max(df[["forward", "turn right", "turn left"]].loc[x])
        self.publishTwist(Learning.twists[action])
        self.reset_world()

        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.1)
            steps += 1

            reward, newState = self.rewardState(Learning.ranges, 0.5, 1.1)
            if state[1] == "far":
                counter += 1
            else:
                counter = 0
            if counter >= 150:
                termination = True
            for d in Learning.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2:
                        reward -= 75
                        termination = True
            if steps >= 800:
                termination = True
            
            state = newState

            x = df[(df["front right"] == state[0]) & (df["front left"] == state[1]) & (df["left"] == state[2]) & (df["back left"] == state[3])].index
            action = max(df[["forward", "turn right", "turn left"]].iloc[x])
            self.publishTwist(Learning.twists[action])
        

    def callback(self, dist):
        Learning.scan = True
        Learning.ranges = list(dist.ranges)
    


if __name__ == "__main__":
    l = Learning(mode="train")
        