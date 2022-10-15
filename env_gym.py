# Third party imports
import gym


class BipedEnv:

    def __init__(self):
        self.env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        self.state_size = self.env.observation_space.shape[0]  # Shape of state array
        self.n_actions = self.env.action_space.shape[0]  # Amount of actions options
        self.max_actions = self.env.action_space.high[0]
        self.min = self.env.action_space.low[0]
        self.max = self.env.action_space.high[0]

        self.record = True
        self.record_eps = 50
        if self.record:
            self.env = gym.wrappers.RecordVideo(
                self.env, 'video', episode_trigger=lambda x: x % self.record_eps == 0)

        print("Size of state array: ", self.state_size)
        print("Amount of actions: ", self.n_actions)
        print(f"Max/Min action: {self.max} - {self.min}")

    def reset_env(self):
        state = self.env.reset()
        return state

    def env_action(self, action):
        next_state, reward, done, extra = self.env.step(action)
        return next_state, reward, done, extra
