import gymnasium
#import random
#from DQN_TF import build_model
#from Agent import build_agent
from keras.optimizers import Adam
import numpy as np
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gymnasium.make("ALE/Freeway-v5", render_mode="human", mode=0) # Gera o ambiente
height, width, channels = env.observation_space.shape
actions = env.action_space.n

#print(env.unwrapped.get_action_meanings()) # Mostra o que cada ação faz

#episodes = 5
# for episode in range(episodes):
#     state, info = env.reset()
#     done = False
#     truncated = False
#     score = 0

#     while not done and not truncated:
#         action = random.choice([0,1,2])
#         n_state, reward, done, truncated, info = env.step(action)
#         score += reward
#     print("Episode: {} Score: {}".format(episode, score))
# env.close()

def build_model(height, width, channels, actions):
    model = Sequential()
    #model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

del model

model = build_model(height, width, channels, actions)
# print(model.summary())

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg', nb_actions=actions, nb_steps_warmup=1000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4))
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)


scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('SavedWeight/10k-Fast/dqn_weights.h5f')
# del model, dqn
# se deletar tem q criar dnv

dqn.load_weights('caminho')