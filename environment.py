import gym
# import Tank
import Navi
import FourRooms
from Traffic.environment_green_wave import *

def OC_env(env_id):
    if env_id in ['Tank-v0', 'Navi-v0','FourRooms-v0']:
        envOC = gym.make(env_id)
    else:
        if env_id in ['TL-v0']:
            envOC = TL_env()
    return envOC
