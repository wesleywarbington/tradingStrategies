"""
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.
"""
  		  	   		  	  		  		  		    	 		 		   		 		  
import random as rand
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		  	  		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		  	  		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		  	  		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		  	  		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		  	  		  		  		    	 		 		   		 		  
        dyna=0,
        verbose=False,
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.num_actions = num_actions  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		  	  		  		  		    	 		 		   		 		  
        self.a = 0

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Qtable = np.zeros((num_states, num_actions))
        self.Ttable = None


    def querysetstate(self, s):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = s

        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # Choose the random direction
        else:
            action = np.argmax(self.Qtable[s,:])

        self.rar = self.rar * self.radr
        self.a = action

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        # Q table update:
        self.Qtable[self.s, self.a] = (1-self.alpha) * self.Qtable[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.Qtable[s_prime,:]))

        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # Choose the random direction
        else:
            action = np.argmax(self.Qtable[s_prime,:])

        # Dyna
        if self.dyna != 0:
            if self.Ttable is None:
                self.Ttable = [[self.s, self.a, s_prime, r]]
            else:
                self.Ttable.append([self.s, self.a, s_prime, r])

            for i in range(self.dyna):
                idx = rand.randrange(0, len(self.Ttable))

                temp_s = self.Ttable[idx][0]
                temp_a = self.Ttable[idx][1]
                temp_s_prime = self.Ttable[idx][2]
                temp_r = self.Ttable[idx][3]

                self.Qtable[temp_s, temp_a] = (1 - self.alpha) * self.Qtable[temp_s, temp_a] + self.alpha * (temp_r + self.gamma * np.max(self.Qtable[temp_s_prime,:]))

        # Update values
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action


