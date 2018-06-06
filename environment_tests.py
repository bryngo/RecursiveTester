from observation_modifier import ObservationModifier

"""
this test is just to make sure that the environment is set up correctly, 
without the need to test through the runner or the agent.
"""
def test_env(env):

    obs_mod = ObservationModifier()

    env.remotes[0].send(("reset", [None]))
    obs = env.remotes[0].recv()

    alt_obs = obs_mod.modify(obs)
