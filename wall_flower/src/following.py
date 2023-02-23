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
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Following():
    twists = {
        'move right': [0.2, -0.95],
        'move left': [0.2, 0.95],
        'forward': [0.25, 0]
    }
    
    scan = None
    ranges = None

    mode = "train"

    startPoses = [(-1.75, 1.8, 0), (1.5, -1, 0), (1.5, 1, 1)]

    def __init__(self, mode="train"):
        Following.mode = mode

        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
        
        if Following.mode == "train":
            self.learning()
        elif Following.mode == "test":
            self.runFile()
        
    def init_services(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        except:
            print("Unable to initiate reset world service")
            raise
        
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        except:
            print("Unable to initiate set model state service")
            raise
        
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        except:
            print("unable to initiate pause physics service")
            raise
        
        rospy.wait_for_service("gazebo/unpause_physics")
        try:
            self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        except:
            print("Unable to initiate unpause physics service")
            raise
    
    def init_node(self):
        rospy.init_node("wallflower")
    
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
    
    def splitRange(self):
        ranges = list(Following.ranges)
        for i,r in enumerate(ranges):
            if r > 3.5:
                ranges[i] = 3.6
        
        front = min([r for r in ranges[324:359]] + [c for c in ranges[0:34]])
        left = min([i for i in ranges[44:134]])
        right = min([j for j in ranges[224:314]])
        return (front, left, right)
    
    def blankTable(self):
        a = {
            "front": [],
            "left": [],
            "right": [],
            "forward": [],
            "move left": [],
            "move right": []
        }

        df = pd.DataFrame(a)

        base_states = ["good", "far"]

        for one in base_states:
            for two in base_states:
                for three in base_states:
                    df.loc[len(df.index)] = [one, two, three, 0, 0, 0]
        
        return df
    
    def updateTable(self, df, state, actions):
        toChange = df[(df["front"] == state[0]) & (df["left"] == state[1]) & (df["right"] == state[2])].index
        df["forward"][toChange] = actions[0]
        df["move left"][toChange] = actions[1]
        df["move right"][toChange] = actions[2]

        return df

    def updateQValue(self, reward, df, old_state, new_state, action, alpha=0.2, gamma=0.8):
        #print(new_state)
        current = df[(df["front"] == old_state[0]) & (df["left"] == old_state[1]) & (df["right"] == old_state[2])]
        updated = df[(df["front"] == new_state[0]) & (df["left"] == new_state[1]) & (df["right"] == new_state[2])]
        actions = current[["forward", "move left", "move right"]]

        m = float(updated[max(list(updated[["forward", "move left", "move right"]]))])
        #print(m)

        newQ = actions[action] + alpha*(reward + gamma*(m - actions[action]))
    #    print("new Q: " + str(float(newQ)))
        df[action][current.index] = float(newQ)
    
    def calculateReward(self, state, mmin, fmin):
        reward = 0

        if state[1] < mmin or state[1] >= fmin or state[0] < mmin or state[2] < mmin:
            reward = -1
        
        return reward
    
    def rewardState(self, mmin, fmin):
        state = self.splitRange()
        reward = self.calculateReward(state, mmin, fmin)
        return reward, state
    
    def episode(self, eps):
        options = [self.makeModelState(p) for p in Following.startPoses]
        try:
            self.set_state(np.random.choice(options))
        except:
            print("A starting state couldn't be found, ending episode...")
            raise

        steps = 0
        counter = 0
        total = 0
        val = 0
        termination = False
        Following.scan = None

        while not Following.scan and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        reward, state = self.rewardState(0.35, 0.55)
        df = self.blankTable()

        x = df[(df["front"] == state[0]) & (df["left"] == state[1]) & (df["right"] == state[2])].index
        
        if random.random() < eps:
            action = np.random.choice(list(df[["forward", "move left", "move right"]].iloc[x]))
        else:
            action = max(df[["forward", "move left", "move right"]].loc[x])
        self.publishTwist(Following.twists[action])

        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.01)
            steps += 1

            reward, newState = self.rewardState(0.35, 0.55)
            self.updateQValue(reward, df, state, newState, action)
            
            if newState[1] >= 0.55:
                counter += 1
            else:
                counter = 0
            if counter >= 200:
                termination = True
            
            for d in Following.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2:
                        reward = -15
                        termination = True
            
            if steps >= 800:
                termination = True
            
            state = newState

            x = df[(df["front"] == state[0]) & (df["left"] == state[1]) & (df["right"] == state[2])].index
            if random.random() < eps:
                action = np.random.choice(list(df[["forward", "move left", "move right"]].iloc[x]))
            else:
                action = max(df[["forward", "move right", "move left"]].iloc[x])
            self.publishTwist(Following.twists[action])
        
        return df
    
    def learning(self):
        eps = 0.9
        duration = 300
        for episode in range(duration):
            print("Episode: " + str(episode))
            updatedDF = self.episode(eps)
            eps = eps - 1.2*(0.8)/duration
        updatedDF.to_json("newData.json")
    
    def runFile(self):
        pass

    def callback(self, dist):
        Following.scan = True
        Following.ranges = list(dist.ranges)


if __name__ == "__main__":
    f = Following(mode="train")