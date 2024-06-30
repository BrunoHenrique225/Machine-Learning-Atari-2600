import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import load_model

# Simplesmente um buffer de memória
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        #self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # Memória que armazena  os estados iniciais
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.uint8)
        #self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32) # a memória de terminal é para saber se é um estado terminal, funciona como uma flag, isso é importante pois o valor do estado terminal é 0
    
    def store_transition(self, state, action, reward, state_, done):
        # cntr = counter
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # sample a memória inteira do agente quando estiver cheia
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) # replace = false faz com que tire a experiencia da lista e não a escolha novamente

        # sample
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# Criando a DQN
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims): # fc = full connected layers
    model = keras.Sequential([keras.layers.Dense(fc1_dims, activation='relu'),
                              keras.layers.Dense(fc2_dims, activation='relu'),
                              keras.layers.Dense(n_actions, activation=None)])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model

# Classe do agente
# Tem a memória, a rede, como todos os hiperparametros dos agentes
# O agente não é uma Deep Q-Network, mas ele a possui
# parei em 10:47
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state,action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            print("Aleatório")
        else:
            print(observation)
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            print(actions)
            action = np.argmax(actions)
            # print("Melhor")
        
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        #15:40
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma + np.max(q_next, axis=1)*dones # é multiplicado pelo dones pois no final vem 0

        #self.q_eval.train_on_batch[states, q_target]
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)

class PreProcessor():
    # Basic constructor
    def __init__(self):
        self = self

    # Convert the RGB 128 color frame to greyscale and store in a compact way
    def toGreyScale(self, frame):
        return np.mean(frame, 2).astype(np.uint8)

    # Downsample the frame by half to reduce the size from 210x160 to 105x80
    def halfDownsample(self, frame):
        # Slices the list in both directions to contain only even indices
        return frame[::2, ::2]

     # Perform a full preprocess on a frame
    def preprocessFrame(self, frame):
        frameGrey = self.toGreyScale(frame)
        processedFrame = self.halfDownsample(frameGrey)
        return processedFrame