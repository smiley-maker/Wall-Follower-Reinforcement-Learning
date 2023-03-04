#!/usr/bin/env python3

#ROS Imports
import sys
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty

#Plotting Imports
import matplotlib
matplotlib.use("Agg") #Uses an interactive backend
import matplotlib.pyplot as plt
import numpy as np
import math

#Data Imports
import json

#Other imports
import make_table


class Learn():
    """
    Class to run Q Learning for wall following robot.
    Includes methods for running Q Learning, testing, 
    and plotting results. 
    """

    #Class variables

    #Twist definitions for each action in the state space
    twists = {
        "right": [0.2, 0.95],
        "left": [0.2, -0.95],
        "forward": [0.2,0.001]
    }

    scan = None #Boolean variable, true if the laser scan just ran
    ranges = None #List of ranges from the robot's laser scan

    mode = "train" #Mode definition to determine whether we are in training or testing mode

    #List of start poses for training
    startPoses = [(-1.75, 1.8, 0.0), (1.5, -1.0, 0.0), (1.5, 1.0, 1.0)]

    def __init__(self, mode="train"):
        """
        Constructor to initialize node, services, publisher, and subscriber. 
        Checks what state the program is in using the mode variable.
        Runs either the Q Learning algorithm or the demo function. 

        Args:
            mode (str, optional): Either "train" or "test" representing the desired mode
        """

        Learn.mode = mode
        self.init_node()
        self.init_services()
        self.init_publisher()
        self.init_subscriber()
        
        if Learn.mode == "train":
            rospy.loginfo('Initializing Training Mode')
            make_table.construct_table()
            self.training()
        elif Learn.mode == "test":
            rospy.loginfo('Initializing Testing Mode')
            self.test()
        else:
            print("Unrecognized mode, please try either train or test.")
    

    def init_services(self):
        """
        Creates the following services
        reset_world: resets the world and all models
        set_model_state: sets the robot's position as desired
        pause_physics: pauses the simulation's physics
        unpause_physics: unpauses the simulation's physics
        """
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        except:
            rospy.loginfo("Unable to initiate reset world service")
            raise
        
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        except:
            rospy.loginfo("Unable to initiate set model state service")
            raise
        
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        except:
            rospy.loginfo("unable to initiate pause physics service")
            raise
        
        rospy.wait_for_service("gazebo/unpause_physics")
        try:
            self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        except:
            rospy.loginfo("Unable to initiate unpause physics service")
            raise

        rospy.loginfo("services set up")
    

    def init_node(self):
        #Initializes a node called wallflower
        rospy.init_node("wallflower")


    def init_publisher(self):
        #Initializes a publisher on the cmd_vel topic to send Twists to the robot. 
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    

    def init_subscriber(self):
        #Initializes the subscriber to take in the robot's laser scan data
        self.sub = rospy.Subscriber("/scan", LaserScan, self.callback)
    

    def publishTwist(self, twister):
        """
        This function creates a Twist object using the input data and publishes
        it based on the previously set up publisher. The input must be a list with
        two values in it: the x velocity and the z angular velocity. 

        Args:
            twister (list): list with twist information for linear x and angular z.
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
        """
        Generates a ModelState message in order to use SetModelState to
        change the robot's starting positions. 

        Args:
            pos (tuple or list): Position to move to. In the format: (x, y, w)

        Returns:
            _type_: _description_
        """
        model_msg = ModelState()

        model_msg.model_name = "turtlebot3_waffle" #Sets the correct name for the turtlebot
        model_msg.pose.position.x = pos[0] #set the x position of the model
        model_msg.pose.position.y = pos[1] #Sets the model's y position
        model_msg.pose.position.z = 0 #Sets the robot's z position to 0
        model_msg.pose.orientation.x = 0
        model_msg.pose.orientation.y = 0
        model_msg.pose.orientation.z = 0
        model_msg.pose.orientation.w = pos[2]
        
        return model_msg
    

    def calculateAccuracy(self, correct, total):
        forwardAccuracy = 0
        leftAccuracy = 0
        rightAccuracy = 0
        #Calculates forward accuracy
        if total[0]: #makes sure that the total is not 0
            forwardAccuracy = correct[0]/total[0]
        #Left accuracy
        if total[1]:
            leftAccuracy = correct[1]/total[1]
        #right accuracy
        if total[2]:
            rightAccuracy = correct[2]/total[2]
        return (forwardAccuracy, leftAccuracy, rightAccuracy)    


    def learningPlot(self, x, y, label, title, color):
        x  = list(range(x)) #Gets the x axis based on the number of episodes

        fig, ax = plt.subplots() #Creates a matplotlib subplot

        #Plots the given data
        ax.plot(x, y, color=color)
        ax.set_title(title)
        
    
        fig.savefig(label)
        plt.close()
    

    def split_range(self):
        """
        Gets the direction of desired ranges by splitting the list
        between left and right.

        Returns:
            List: List containing approximate front and left distances
        """
        for i,r in enumerate(Learn.ranges):
            if r > 3.5:
                Learn.ranges[i] = 3.6
        
        front = min([f for f in Learn.ranges[0:44] + [o for o in Learn.ranges[314:359]]])
        left = min([l for l in Learn.ranges[44:134]])

        return [front, left]
    

    def get_table(self, filename):
        """
        Gets a stored q table from a json file. 

        Args:
            filename (string): file name where the q table is stored

        Returns:
            Dictionary: Q table with state, action values
        """
        with open(filename, "r") as f:
            q = json.load(f)
        return q


    def save_table(self, table, filename):
        """
        Saves a q table in dictionary format as a json file

        Args:
            table (dictionary): Q table formatted as a dictionary
            filename (string): name of the file desired
        """
        with open(filename, "w") as f:
            json.dump(table, f, indent=4)
    

    def getStringState(self, state, mmin, fmin):
        """
        Generates a string version of the input state based on
        different ranges: 0-cmin is close, mmin-fmin is medium,
        and fmin-4 is far. 

        Args:
            state (tuple): tuple containing front and left values,
            mmin (float): minimum medium value
            fmin (float): _minimum far threshold

        Returns:
            _: string representing numerical state
        """
        res = "" #Variable to store the string representation of the input state
        
        if 0 <= state[0] < mmin:
            res += "forward: close, "
        elif mmin <= state[0] < fmin:
            res += "forward: medium, "
        elif fmin <= state[0] <= 4:
            res += "forward: far, "

        if 0 <= state[1] < mmin:
            res += "left: close"
        elif mmin <= state[1] < fmin:
            res += "left: medium"
        elif fmin <= state[1] <= 4:
            res += "left: far"

        return res
    

    def calculateReward(self, state, mmin, fmin):
        #Calculates the reward based on the following simple guidelines
        reward = 0
        #If the left side is too close or too far from the wall, or if the forward side is too close,
        if state[1] < mmin or state[1] >= fmin or state[0] < mmin:
            reward = -1 #reward is -1. 
        
        return reward
    

    def rewardState(self, mmin, fmin):
        #Helper function to calculate the reward and current state
        state = self.split_range()
        reward = self.calculateReward(state, mmin, fmin)
        state = self.getStringState(state, mmin, fmin)
        return reward, state
    

    def updateQValue(self, reward, Q, old_state, new_state, action, new_action=None, strategy="sarsa", alpha=0.1, gamma=0.9):
        """
        This function is used by the general Q Learning/SARSA algorithm below. It updates
        the Q table using the chosen strategy. 

        Args:
            reward (integer): reward value obtained for old action
            Q (_type_): current Q table
            old_state (string): previous state (left, right string)
            new_state (string): next state (left, right string)
            action (string): Action taken for previous state
            new_action (string, optional): If using sarsa, a new action must be included for the new state. Defaults to None.
            strategy (string, optional): String representing the strategy desired. Current options are sarsa or temporal difference. Defaults to "sarsa".
            alpha (float, optional): Weight of temporal difference. Defaults to 0.1.
            gamma (float, optional): weight of new state. Defaults to 0.9.
        """
        if strategy == "sarsa" and new_action:
            temp_diff = reward + gamma*((Q[new_state][new_action]))
        else:
            temp_diff = reward + gamma*(max(Q[new_state].values()))
        
        Q[old_state][action] = Q[old_state][action] + alpha*(temp_diff - Q[old_state][action])
    

    def episode(self, Q, eps):
#        self.pause_physics() #pauses physics in the simulator while calculating
        steps = 0 #Variable to store the number of steps in the episode
        counter = 0 #Counts the number of times in a row that the robot has a left wall in the far range. 
        forwardTotal = 0 #Total number of times the robot was in a set of given states
        forwardCorrect = 0 #Number of times the robot performed the correct action while in forward state
        leftTotal = 0 #Total number of times the robot was in a set of left states
        leftCorrect = 0 #Counter for the number of times the robot acted correctly when in a left state
        rightTotal = 0 #Total number of times the robot was in a set of right states
        rightCorrect = 0 #Counter for the number of times the robot acted correctly when in a right state
        forwardAccuracy = 0
        leftAccuracy = 0
        rightAccuracy = 0
        termination = False
        totalReward = 0

        options = [self.makeModelState(pos) for pos in self.startPoses]
        self.set_state(np.random.choice(options))
        self.publishTwist([0,0]) #Publishes a twist with 0 linear/angular velocities

        Learn.scan = None #Sets the scan variable to none when starting an episode (waits for the next scan)
        while not Learn.scan and not rospy.is_shutdown():
            rospy.sleep(0.1) #waits for valid scan data
        
        reward, state = self.rewardState(0.5, 0.75) #Gets the current state and reward using a good range of 0.5-0.75

        #Takes an action
        #random action
        if np.random.random() < eps:
            action = np.random.choice(list(Q[state].keys())) #Makes a random choice from the current state's actions
        else: #motivated choice
            action = max(Q[state], key=Q[state].get) #chooses the action with the highest reward for the current state
        
 #       self.unpause_physics() #unpauses physics to publish and see result
        self.publishTwist(Learn.twists[action]) #Publishes a twist using the current action

        #Main loop (runs until episode is terminated)
        while not termination and not rospy.is_shutdown():
#            self.pause_physics() #pauses physics while calculating
            rospy.sleep(0.1) #Sleeps for 0.1 seconds (since the LaserScan publishes at 10 Hz)

            steps += 1 #increments the total number of steps taken
            
            reward, newState = self.rewardState(0.5, 0.75) #Gets the new state and reward using a good range of 0.5-0.75
            totalReward += reward

            #Termination checks ->

            #Increments the counter if there is no left wall in the medium range
            if "left: far" in newState:
                counter += 1
            else:
                counter = 0
            #If the robot went 250 steps in a row without being near a left wall,
            if counter == 250:
                forwardAccuracy, leftAccuracy, rightAccuracy = self.calculateAccuracy([forwardCorrect, leftCorrect, rightCorrect], [forwardTotal, leftTotal, rightTotal])
                termination = True
                break
            
            #Loops through the ranges from the laser scan and checks if any of them are really close to a wall
            for d in Learn.ranges:
                if not rospy.is_shutdown():
                    if d < 0.2:
                        reward = -15
                        forwardAccuracy, leftAccuracy, rightAccuracy = self.calculateAccuracy([forwardCorrect, leftCorrect, rightCorrect], [forwardTotal, leftTotal, rightTotal])
                        termination = True
                        break
            
            #If the episode has lasted 800 steps,
            if steps >= 800:
                forwardAccuracy, leftAccuracy, rightAccuracy = self.calculateAccuracy([forwardCorrect, leftCorrect, rightCorrect], [forwardTotal, leftTotal, rightTotal])
                steps = 0
                termination = True
                break

            #Decisions
            #random action
            if np.random.random() < eps:
                new_action = np.random.choice(list(Q[newState].keys()))
            else: #actual decision
                new_action = max(Q[newState], key=Q[newState].get)
                
                #Checks if the state is one of the states I considered to have actions of forward, right, and left, respectively
                #Change before submitting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if state == "forward: medium, left: medium":
                    leftTotal += 1
                    if new_action == "right":
                        leftCorrect += 1
                
                if state == "forward: far, left: medium":
                    forwardTotal += 1
                    if new_action == "forward":
                        forwardCorrect += 1
                
                if state == "forward: far, left: far":
                    rightTotal += 1
                    if new_action == "left":
                        rightCorrect += 1
            

 #           self.unpause_physics() #unpauses physics while publishing
            #Publishes a twist with the chosen action
            self.publishTwist(self.twists[new_action])
            #Updates the q table
            self.updateQValue(reward, Q, state, newState, action, new_action=new_action, strategy="td")
            #sets the variables for the next iteration
            action = new_action
            state = newState
        
        #Once the episode finishes, the total and correct values for each action are returned for plotting
        return [(forwardAccuracy, forwardTotal), (rightAccuracy, rightTotal), (leftAccuracy, leftTotal), totalReward]
    

    def training(self):
        """
        Runs multiple episodes to determine an accurate Q table and plot relevant data.
        """

        #Variables
        eps = 0.9 #epsilon used to determine chance of random action
        duration = 500 #Number of episodes run in a training session
        rightData = []
        leftData = []
        forwardData = []
        rewardData = []
        rightEpisodes = 0
        leftEpisodes = 0
        forwardEpisodes = 0

        #Loops through the length of the duration
        for iteration in range(duration):
            if not rospy.is_shutdown():
                #Prints the episode for reference
                rospy.loginfo("Episode: " + str(iteration))

                #Gets the current q table from the stored file
                Q = self.get_table("minimalQ3.json")
                #Runs the episode and gets the accuracy data
                accData = self.episode(Q, eps)
                #Forward distribution:
                if accData[0][1] >= 15: #I decided to reduce noise in my graphs by only taking episodes with at least 20 steps
                    forwardEpisodes += 1
                    forwardData.append(accData[0][0]) #Appends the current forward accuracy
                
                #Right distribution:
                if accData[1][1] >= 15:
                    rightEpisodes += 1
                    rightData.append(accData[1][0]) #Appends the current right accuracy
                
                #Left distribution
                if accData[2][1] >= 15:
                    leftEpisodes += 1
                    leftData.append(accData[2][0]) #Appends the current left accuracy

                rewardData.append(accData[3])
                
                #Plots the current learning trends for each action
                self.learningPlot(forwardEpisodes, forwardData, "TDforwardLearning.png", "TD Forward", "#50b6fa")
                self.learningPlot(rightEpisodes, rightData, "TDrightLearning.png", "TD Right", "#ff4de4")
                self.learningPlot(leftEpisodes, leftData, "TDleftLearning.png", "TD Left", "#a04dff")
                self.learningPlot(iteration+1, rewardData, "RewardTD.png", "TD Reward", "#a04dff")

                #Saves the q table to a json file
                self.save_table(Q, "minimalQ2.json")

                #Reduces epsilon (It should eventually move from 0.9 to 0.1)
                eps -= 1.2*(0.8/duration)
    

    def runFile(self, Q):
        Learn.scan = None #Sets the current scan data to none so that the algorithm can start with latest scan
        termination = False
        steps = 0

  #      self.unpause_physics()

        options = [self.makeModelState(pos) for pos in self.startPoses]
        self.set_state(np.random.choice(options))
        self.publishTwist([0,0]) #Publishes a twist with 0 linear/angular velocities


        while not rospy.is_shutdown() and not Learn.scan:
            rospy.sleep(0.1) #waits for scan daata
        
        #Gets initial state and reward
        reward, state = self.rewardState(0.5, 0.75)

        #Gets greedy action (no randomness when simply testing)
        action = max(Q[state], key=Q[state].get)

        #Publishes twist corresponding to the action chosen
        self.publishTwist(Learn.twists[action])

        #Loops until terminated
        while not termination and not rospy.is_shutdown():
            rospy.sleep(0.1)
 #           self.pause_physics() #pauses physics in the simulation while calculating
#            rospy.sleep(0.1)
            steps += 1 #Increments the steps with each iteration

            #Calculates the reward and new state
            reward, newState = self.rewardState(0.5, 0.75)

            #Checks to see if the robot has crashed
            for d in Learn.ranges:
                if not rospy.is_shutdown():
                    if d < 0.15:
                        termination = True
                        break
            
            #Checks if the robot has completed 800 steps (end goal)
            if steps >= 800:
                termination = True
                break
            
            #Updates the state for the next iteration
            state = newState

            #Gets the action for the current state using a purely greedy choice
            action = max(Q[state], key=Q[state].get)

#            self.unpause_physics() #Unpauses physics to publish
            #Publishes a twist corresponding to the calculated action
            self.publishTwist(self.twists[action])
    

    def test(self):
        duration = 100 #Demo duration

        q = self.get_table("minimalQ3.json")

        for e in range(duration): #loops through the duration
            if not rospy.is_shutdown():
                print(e)
                self.runFile(q)
                rospy.sleep(0.1)
    

    def callback(self, dist):
        #callback for the subscriber that gets the laser scan data and sets class variables
        Learn.scan = True
        Learn.ranges = list(dist.ranges)



if __name__ == "__main__":
    
    l = Learn(mode="train")