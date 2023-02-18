#!/usr/bin/env python3

#Imports
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import pandas as pd
from std_srvs.srv import Empty
import json

class makeTable():
    q = {} #Queue table
    states = [] #List of possible states
    actions = {} #Dictionary of actions and the best option (all initialized to 0)
    def __init__(self, states, actions): 
        """
        Class constructor. Sets the states and actions. 

        Args:
            states (List): List of possible states for this system
            actions (Dictionary): Dictionary of actions and their corresponding utilities. 
        """        
        self.states = states
        self.actions = actions
    
    def formatDict(self):
        """
        Formats the dictionary to contain all of the action values for a single state. 

        Returns:
            Dictionary: Formatted queue table
        """        
        for state in self.states:
            self.q[state] = self.actions
        return self.q
    
    def saveToJson(self):
        """
        Generates a json file using the queue table determined above. 
        """        
        with open("wall_flower/src/CurrentQ.json", "w") as f:
            json.dump(self.q, f, indent=4)
    


#Sets up the actions and states I chose to use. 
actions = {"forward": 0, "right": 0, "left": 0}
states = []
base_states = ["close", "medium", "far"]
#Loops through the base state three times to create a string containing
#each of forward, right, and left. 
for one in base_states:
    for two in base_states:
        for three in base_states:
            state = f"forward: {one}, right: {two}, left: {three}"
            states.append(state)

#Makes, formats, and saves the queue table. 
tableMaker = makeTable(states, actions)
tableMaker.formatDict()
tableMaker.saveToJson()
