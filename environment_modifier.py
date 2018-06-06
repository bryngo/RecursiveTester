class EnvironmentModifier:

    # Modifies the observations and rewards, takes in arrays of n_envs length (not known)
    def modify(self, observations, rewards, last_observations):
        new_observations = {}
        new_rewards = {}
        return new_observations, new_rewards
