from multiprocessing import Process, Pipe
from pysc2.env import sc2_env
from pysc2.env.sc2_env import Agent, Bot
from functools import partial
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['environment.py'])

_TERRAN = 2
_ZERG = 3

#references:
#   https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
#   https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/environment.py
#   https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
#       other pysc2 git repos too, but above is the main one


class Environment:

    def __init__(self, n_envs=1):
        self.n_envs = n_envs

        env_args = self.getArgs()

        # using args, create callable functions for game initialization for each worker in pipe
        env_fns = [partial(make_sc2env, **env_args)] * n_envs

        # create pipe of workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def getArgs(self):
        """
        set up arguments to give to sc2_env upon creation
        """
        agent_type = Agent(_ZERG) # specify agent race
        bot_type = Bot(_TERRAN, 1) # specify bot race and difficulty
        player_types = [agent_type, bot_type]

        agent_interface = sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            action_space="FEATURES", # actions.ActionSpace.FEATURES,
            use_feature_units=True)

        env_args = dict(
            map_name="Simple64",
            step_mul=8,
            game_steps_per_episode=0,
            score_index=-1,
            disable_fog=True,
            agent_interface_format=agent_interface,
            players=player_types)

        # add visualization if running only one game environment
        if self.n_envs == 1:
            env_args['visualize'] = True

        return env_args

def worker(remote, env_fn_wrapper):
    """
    A worker is like an instance of the game.
    """
    env = env_fn_wrapper.x()
    while True:
        cmd, action = remote.recv()
        if cmd == 'step':
            timesteps = env.step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'reset':
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'observation_spec':
            spec = env.observation_spec()
            remote.send(spec)
        elif cmd == 'action_spec':
            spec = env.action_spec()
            remote.send(spec)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries
    to use pickle).
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def make_sc2env(**kwargs):
    """
    function called by each environment when created in pipe
    when called, **kwargs comes from getArgs() function
    """
    env = sc2_env.SC2Env(**kwargs)
    return env