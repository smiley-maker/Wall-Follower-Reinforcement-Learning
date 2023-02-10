#!/usr/bin/env python3

#Imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import pandas as pd
from std_srvs.srv import Empty
import json

class makeTable():
    q = {}
    states = []
    actions = {}
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
    
    def formatDict(self):
        for state in self.states:
            self.q[state] = self.actions
        return self.q
    
    def saveToJson(self):
        with open("QQ.json", "w") as f:
            json.dump(self.q, f, indent=4)
    
    def loadJson(self):
        with open("QQ.json", "r") as f:
            qDict = json.load(f)
        return qDict


actions = {"forward": 0, "right": 0, "left": 0}
states = []
base_states = ["close", "medium", "far"]
for one in base_states:
    for two in base_states:
        for three in base_states:
            state = f"forward: {one}, right: {two}, left: {three}"
            states.append(state)

tableMaker = makeTable(states, actions)
tableMaker.formatDict()
tableMaker.saveToJson()
