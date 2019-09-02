import gym
import numpy as np
import copy
import random
import math
from gym import spaces, error


class NaviEnv(gym.Env):

    def __init__(self):

        self.State= np.array([0,0,0,0,0])
        # Checks if the agent got the trap candy at state -2
        self.GotCandyReward = False

        # The actions are either
        # 0 -> move towards the -ve branch key (discounted)
        # 1 -> move towards the +ve branch key (average)
        self.action_space = spaces.Discrete(2)


        # The space has states from -5 to 5
        # The first dim is the branch either -1 (discounted branch), 0 Start State, +1 (average branch)
        # The 2nd, 3rd, 4th dims indicate a binary number from 1-5
        # The 5th dim indicates which key the agent got -1 -> DR key, 0 -> no key or 1 ->AR key
        self.observation_space = spaces.Box( np.array([-1,0,0,0,-1]), np.array([0,+1,+1,+1,+1]), dtype=np.int32)
        self.reset()

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        self.State= np.array([0,0,0,0,0])
        self.GotCandyReward = False
        return self._get_game_state()

    def _get_game_state(self):
        return copy.deepcopy(self.State)

    def render(self):
        """
        This function renders the current game state in the given mode.
        """
        StateDecimalValue, key = self._StateToDecimal(self.State)
        if(key==0):
            OutputVal="The state is " + str(StateDecimalValue) +" MISSED the key"
        else:
            OutputVal="The state is "+ str(StateDecimalValue) +" Got the key"
        return OutputVal


    def _StateToDecimal(self,binaryValue):
        DecValue=0
        # the for loop converts binary to decimal
        for i in range(3):
            DecValue=DecValue*2
            DecValue+=binaryValue[i+1]

        # To get the sign
        if(binaryValue[0]== -1):
            DecValue=DecValue*binaryValue[0]
        return DecValue, binaryValue[-1]

    def _is_over(self):
        toCheck=self._get_game_state()
        StateDecimalValue, key = self._StateToDecimal(toCheck)
        if(StateDecimalValue==0):
            return True
        else:
            return False

    def _DecimalToState(self,DecimalValue, key):
        binarySt= np.array([0,0,0,0,0])

        binarySt[-1] = key
        if(DecimalValue < 0):
            binarySt[0] = -1
        # the for loop converts binary to decimal
        absDec=abs(DecimalValue)
        if(absDec % 2 == 1):
            binarySt[3] = 1
        absDec=int(absDec/2)
        if(absDec % 2 == 1):
            binarySt[2] = 1
        absDec=int(absDec/2)
        if(absDec % 2 == 1):
            binarySt[1] = 1
        absDec=int(absDec/2)
        # print("this is the binary state",binarySt)
        return binarySt


    def step(self, action):
        """
        Parameters
        ----------
        action : int
            The action is between 0 and 1
            decides the direction of the navigation.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        St = self._get_game_state()
        DecimalSt, key = self._StateToDecimal( St )
        # print("this is the state in step",DecimalSt, key)
        if(action==0):
            if(DecimalSt>-5):
                DecimalSt-=1
                if(DecimalSt==-5):
                    key=-1
        else:
            if(DecimalSt<5):
                DecimalSt+=1
                if(DecimalSt==5):
                    key=1

        # print("this is the state in step",DecimalSt, key)
        self.State=self._DecimalToState(DecimalSt, key)
        Gamedone = self._is_over()
        newState = self._get_game_state()
        reward = self._get_reward()
        return newState, reward, Gamedone, {}

    def _get_reward(self):
        """
        This function calculates the reward.
        """
        St = self._get_game_state()
        DecimalSt, key = self._StateToDecimal(St)
        if(DecimalSt == 0):
            if(key==1):
                return 12.0
            else:
                if(key==-1):
                    return 10.0
                else:
                    return 0.0
        else:
            if(DecimalSt == -2 and self.GotCandyReward==False):
                self.GotCandyReward=True
                return 1.0
            else:
                return 0.0
