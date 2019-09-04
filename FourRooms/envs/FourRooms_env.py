import gym
import numpy as np
import copy
import random
import math
from gym import spaces, error


class FourRoomsEnv(gym.Env):

    def __init__(self, pr=0.75):
        # pr is the probability a transition will succeed
        self.xBits=2 # x1,x2
        self.yBits=2 # y1,y2
        self.p=pr

        self.gotBasementReward=False
        self.gotKeyB=False
        self.gotKeyR=False


        # state bits division (9 bits)
        # x1 co-ord, x2 co-ord, y1 co-ord, y2 co-ord, Basement, Ground, Roof, KeyB, KeyR
        self.State= np.array([0,0,0,0, 0,1,0, 0,0])
        # Checks if the agent got the trap candy at state -2

        # The actions are either
        # Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)


        self.observation_space = spaces.Box( np.array([0,0,0,0, 0,0,0, 0,0]), np.array([+1,+1,+1,+1, +1,+1,+1, +1,+1]), dtype=np.int32)
        self.reset()

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        self.State= np.array([0,0,0,0, 0,1,0, 0,0])
        self.gotBasementReward=False
        self.gotKeyB=False
        self.gotKeyR=False
        return self._get_game_state()

    def _get_game_state(self):
        return copy.deepcopy(self.State)

    def render(self):
        """
        This function renders the current game state in the given mode.
        """
        TempState=self._get_game_state()
        print("#########################")
        if(TempState[ self.xBits+self.yBits ]==1):
            print("In the Basement")
        elif(TempState[ self.xBits+self.yBits+1 ]==1):
            print("On the Ground Floor")
        elif(TempState[ self.xBits+self.yBits+2 ]==1):
            print("On the Roof")

        if(TempState[ self.xBits+self.yBits+3 ]==1):
            print("Has Key B")
        elif(TempState[ self.xBits+self.yBits+4 ]==1):
            print("Has Key R")

        Xcoord = self._XorYToDecimal(TempState[0:self.xBits])
        Ycoord = self._XorYToDecimal(TempState[2:2+self.yBits])

        for i in reversed(range( int(math.pow(2,self.yBits)) )):
            # for each row we will print a string
            rowToRender=""
            for j in range(int( math.pow(2,self.xBits) )):
                if(i!=Ycoord):
                    rowToRender="_"*int( math.pow(2,self.xBits) )
                else:
                    if(j==Xcoord):
                        rowToRender=rowToRender+"A"
                    else:
                        rowToRender=rowToRender+"_"
            print(rowToRender)

        print("#########################")


    def _XorYToDecimal(self,binaryValue):
        DecValue=0
        # the for loop converts binary to decimal
        for i in range(self.xBits):
            DecValue=DecValue*2
            DecValue+=binaryValue[i]
        return DecValue

    def _add1toXYBinary(self,binaryValue,plus):
        # the for loop converts binary to decimal
        for i in reversed(range(self.xBits)):
            if plus:
                if(binaryValue[i]==0):
                    binaryValue[i]=1
                    break
                else:
                    binaryValue[i]=0

            else:
                if(binaryValue[i]==1):
                    binaryValue[i]=0
                    break
                else:
                    binaryValue[i]=1
        return binaryValue

    def _is_over(self):
        toCheck=self._get_game_state()
        if(toCheck[self.xBits+self.yBits+1]==1):
            # We are on the ground floor
            if( self._XorYToDecimal(toCheck[0:self.xBits])==0 ):
                # x coordinate is 0
                if( self._XorYToDecimal(toCheck[2:2+self.yBits])==0 ):
                    # y coordinate is 0
                    return True

        return False

    def _reachedKey(self, St):
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        if(St[ self.xBits+self.yBits+3 ]==0 and St[ self.xBits+self.yBits+4 ]==0):

            if( St[ self.xBits+self.yBits ]==1 and Xcoord==3 and Ycoord==3):
                # Basement & X==3 & Y==3
                St[ self.xBits+self.yBits+3 ]=1

            elif(St[ self.xBits+self.yBits+2 ]==1 and Xcoord==3 and Ycoord==3):
                # Roof & X==3 & Y==3
                St[ self.xBits+self.yBits+4 ]=1

        return St

    def _jumpingFloors(self, St, action):
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        if(St[ self.xBits+self.yBits+1 ]==1 and Xcoord==1 and Ycoord==3 and action==0):
            # Ground & X==1 & Y==3 & action=0
            return True

        elif(St[ self.xBits+self.yBits+1 ]==1 and Xcoord==3 and Ycoord==1 and action==3):
            # Ground & X==3 & Y==1 & action=3
            return True

        elif(St[ self.xBits+self.yBits+3 ]==1 and St[ self.xBits+self.yBits ]==1 and Xcoord==1 and Ycoord==0 and action==1):
            # KeyB & Basement & X==1 & Y==0 & action=1
            return True

        elif(St[ self.xBits+self.yBits+4 ]==1 and St[ self.xBits+self.yBits+2 ]==1 and Xcoord==0 and Ycoord==1 and action==2):
            # KeyR & Roof & X==0 & Y==1 & action=2
            return True

        return False

    def _DoorTravel(self, St, action):
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        if(St[ self.xBits+self.yBits+1 ]==1 and Xcoord==1 and Ycoord==3 and action==0):
            # Ground & X==1 & Y==3 & action=0
            if(np.random.random()<self.p):
                # Go from ground to Basement
                St[ self.xBits+self.yBits+1 ]=0
                St[ self.xBits+self.yBits ]=1

                for i in range(self.yBits):
                    # make Ycoord =0
                    St[self.xBits+i]=0

        elif(St[ self.xBits+self.yBits+1 ]==1 and Xcoord==3 and Ycoord==1 and action==3):
            # Ground & X==3 & Y==1 & action=3
            if(np.random.random()<self.p):
                # Go from ground to Roof
                St[ self.xBits+self.yBits+1 ]=0
                St[ self.xBits+self.yBits+2 ]=1

                for i in range(self.xBits):
                    # make Xcoord =0
                    St[i]=0

        elif(St[ self.xBits+self.yBits+3 ]==1 and St[ self.xBits+self.yBits ]==1 and Xcoord==1 and Ycoord==0 and action==1):
            # KeyB & Basement & X==1 & Y==0 & action=1
            if(np.random.random()<self.p):
                # Go from ground to Basement
                St[ self.xBits+self.yBits ]=0
                St[ self.xBits+self.yBits+1 ]=1

                for i in range(self.yBits):
                    # make Ycoord =0
                    St[self.xBits+i]=1

        elif(St[ self.xBits+self.yBits+4 ]==1 and St[ self.xBits+self.yBits+2 ]==1 and Xcoord==0 and Ycoord==1 and action==2):
            # KeyR & Roof & X==0 & Y==1 & action=2
            if(np.random.random()<self.p):
                # Go from Roof to Ground
                St[ self.xBits+self.yBits+2 ]=0
                St[ self.xBits+self.yBits+1 ]=1

                for i in range(self.xBits):
                    # make Xcoord =0
                    St[i]=1

        return St

    def _move(self, St, action):
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        # The actions are Up, Down, Left, Right

        if(action==0):
            if (Ycoord<int( math.pow(2,self.yBits) )-1 ):
                # not on upper most row
                if(np.random.random()<self.p):
                    St[2:2+self.yBits]=self._add1toXYBinary(St[2:2+self.yBits], True)

        elif(action==1):
            if (Ycoord>0 ):
                # not on bottom most row
                if(np.random.random()<self.p):
                    St[2:2+self.yBits]=self._add1toXYBinary(St[2:2+self.yBits], False)

        elif(action==2):
            if (Xcoord>0 ):
                # not on left most row
                if(np.random.random()<self.p):
                    St[0:self.xBits]=self._add1toXYBinary(St[0:self.xBits], False)

        elif(action==3):
            if (Xcoord<int( math.pow(2,self.yBits) )-1 ):
                # not on right most row
                if(np.random.random()<self.p):
                    St[0:self.xBits]=self._add1toXYBinary(St[0:self.xBits], True)

        return St


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
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        if( self._jumpingFloors(St, action) ):
            St=self._DoorTravel(St, action)
        else:
            # moving on a floor
            St=self._move(St,action)

        St=self._reachedKey(St)

        self.State=St
        Gamedone = self._is_over()
        newState = self._get_game_state()
        reward = self._get_reward()
        return newState, reward, Gamedone, {}

    def _get_reward(self):
        """
        This function calculates the reward.
        """
        St = self._get_game_state()
        Xcoord = self._XorYToDecimal(St[0:self.xBits])
        Ycoord = self._XorYToDecimal(St[2:2+self.yBits])

        if(St[ self.xBits+self.yBits ]==1 and not self.gotBasementReward):
            # got to the Basement
            self.gotBasementReward=True
            return 1.0

        elif(St[ self.xBits+self.yBits+3 ]==1 and not self.gotKeyB):
            self.gotKeyB=True
            return 1.0

        elif(St[ self.xBits+self.yBits+4 ]==1 and not self.gotKeyR):
            self.gotKeyR=True
            return 1.0

        elif( St[ self.xBits+self.yBits+3 ]==1 and Xcoord==0 and Ycoord==0 and St[ self.xBits+self.yBits+1 ]==1):
            return 5.0

        elif( St[ self.xBits+self.yBits+4 ]==1 and Xcoord==0 and Ycoord==0 and St[ self.xBits+self.yBits+1 ]==1):
            return 10.0

        else:
            return 0.0
