import gymnasium
from gymnasium.wrappers import FlattenObservation
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from keras import layers

from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
env = gymnasium.make("ALE/Boxing-v5", obs_type="grayscale")
env = FlattenObservation(env)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space.shape))

train_episodes = 675
test_episodes = 100

class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=object)
        self.index = 0
        self.length = 0
        
    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
    
    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)
# qooapp
#====================================
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden = 512
initializer = keras.initializers.VarianceScaling()
hidden_activation = tf.nn.relu
def agent(state_shape, action_shape):    
    learning_rate = 0.001
    init = tf.keras.initializers.VarianceScaling()

    model = keras.Sequential()
    model.add(layers.Input(shape=state_shape))

    # Reshape a entrada para o formato adequado para Conv2D (por exemplo, 210x160x1 para Atari)
    model.add(layers.Reshape((210, 160, 1)))  # Ajuste este reshape com base nas dimensões corretas da sua observação original

    # Adicionando camadas convolucionais
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding="same", activation='relu', kernel_initializer=init))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding="same", activation='relu', kernel_initializer=init))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer=init))
    
    # Achatar os dados para usar camadas densas
    model.add(layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer=init))

    # Camada de saída linear para prever ações
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))

    model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.001 # Learning rate
    discount_factor = 0.99

    MIN_REPLAY_SIZE = 10000
    #if len(replay_memory) < MIN_REPLAY_SIZE:
    if replay_memory.length < MIN_REPLAY_SIZE:
        return

    batch_size = 50
    mini_batch = replay_memory.sample(batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        #current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q
        current_qs[action] = reward + discount_factor * np.max(future_qs_list[index]) if not done else reward

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    
def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the timee
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.0012 # Depende do número de episódios 
    eps_decay_steps = 675

    model = agent(env.observation_space.shape, env.action_space.n)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    #replay_memory = deque(maxlen=1_000_000)

    target_update_counter = 0

    X = []
    y = []

    rewards = []
    episodes = []
    punches = []
    recieved = []

    steps_to_update_target_model = 0
    #==================================
    for episode in range(train_episodes):
        print(episode)
        total_training_rewards = 0
        total_training_punches = 0
        total_training_losses = 0
        optimal_action = 0
        random_action = 0
        observation, info = env.reset(seed=RANDOM_SEED)    
        done = False
        episodes.append(episode + 1)
        while not done:    
            steps_to_update_target_model += 1

            random_number = np.random.rand()
            if random_number <= epsilon:
                action = env.action_space.sample()
                random_action += 1
            else:
                encoded = observation  
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()                
                action = np.argmax(predicted)
                optimal_action += 1
                
            new_observation, reward, done, truncated, info = env.step(action)            
            if reward < 0: total_training_losses += 1
            if reward > 0: total_training_punches += 1

            replay_memory.append([observation, action, reward, new_observation, done])
            #replay_memory.append((observation, action, reward, new_observation, 1.0 - done))

            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            if steps_to_update_target_model % 10000 == 0 or done:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1
                epsilon = max(min_epsilon, max_epsilon - (max_epsilon-min_epsilon) * episode/eps_decay_steps)
                break
        #======================================
        f = open("Resume.txt", "a")
        f.write("Episode: " + str(episode) + "\n")
        f.write("Total Recompensas: " + str(total_training_rewards) + "\n")
        f.write("Total Socos Dados: " + str(total_training_punches) + "\n")
        f.write("Total Socos Recebidos: " + str(total_training_losses) + "\n")
        f.write("Acoes Otimas: " + str(optimal_action) + "\n")
        f.write("Acoes Aleatorias: " + str(random_action) + "\n")
        f.write("Ypsolon: " + str(epsilon) + "\n")
        now = datetime.now()
        dt_string = now.strftime("%d/%m %H:%M:%S \n")
        f.write(dt_string)
        f.write("==========" + "\n")
        f.close() 
        rewards.append(total_training_rewards)
        punches.append(total_training_punches)
        recieved.append(total_training_losses)

        #epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        #break
    env.close()
    plt.plot(episodes, rewards, label = "Reward", linestyle="-") 
    plt.plot(episodes, punches, label = "Punches", linestyle="--") 
    plt.plot(episodes, recieved, label = "Losses", linestyle="-.") 
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa')
    plt.title("Episódio x Recompensa")
    plt.legend()
    plt.show()
    # Será exibido respectivamente, os episódios, recompensas, acertos do nosso personagem e do inimigo
    f = open("Resume.txt", "a")    
    f.write("Total Recompensas: ")
    for recompensa in rewards:
        f.write(str(recompensa) + ", ")
    f.write("\n")
    f.write("Total Socos: ")
    for socos in punches:
        f.write(str(socos) + ", ")
    f.write("\n")
    f.write("Total Socos Recebidos: ")
    for recebidos in recieved:
        f.write(str(recebidos) + ", ")
    f.close() 

if __name__ == '__main__':
    main()