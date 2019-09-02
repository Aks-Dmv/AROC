from gym.envs.registration import register

register(
    id='FourRooms-v0',
    entry_point='FourRooms.envs:FourRoomsEnv'
)
