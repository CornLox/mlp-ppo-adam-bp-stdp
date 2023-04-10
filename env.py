import gym
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# env setup
class Enviroment(gym.vector.SyncVectorEnv):
    def __init__(self,args,run_name):
        super(Enviroment,self).__init__([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
    def check_support(self):
        assert isinstance(self.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


      