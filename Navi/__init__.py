from gym.envs.registration import register

register(
    id='Navi-v0',
    entry_point='Navi.envs:NaviEnv'
)
