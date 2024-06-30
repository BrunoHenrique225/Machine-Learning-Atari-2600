import cv2

import gymnasium
import gymnasium.spaces

import numpy as np

class ProcessFrame84(gymnasium.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ScaledFloatFrame(gymnasium.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# class BufferWrapper(gymnasium.ObservationWrapper):
#     def __init__(self, env, n_steps, dtype=np.float32):
#         super(BufferWrapper, self).__init__(env)
#         self.dtype = dtype
#         old_space = env.observation_space
#         self.observation_space = gymnasium.spaces.Box(old_space.low.repeat(n_steps, axis=0),
#                                                 old_space.high.repeat(n_steps, axis=0), dtype=dtype)

#     def reset(self):
#         self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
#         return self.observation(self.env.reset())

#     def observation(self, observation):
#         self.buffer[:-1] = self.buffer[1:]
#         self.buffer[-1] = observation
#         return self.buffer

class BufferWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gymnasium.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env():
    env = gymnasium.make("ALE/Freeway-ram-v5", render_mode="human", mode=0)
    env = ProcessFrame84(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)