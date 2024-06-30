from DQN_Freeway import Agent
from DQN_Freeway import PreProcessor
import numpy as np
import gymnasium
import tensorflow as tf

preProcessor = PreProcessor()

if __name__ == "__main__":
    #tf.compat.v1.disable_eager_execution()
    
    env = gymnasium.make("ALE/Freeway-ram-v5", render_mode="human", mode=7)
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space.shape, n_actions=env.action_space.n, mem_size=1000000, batch_size=64, epsilon_end=0.01)    
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation, info = env.reset()
        print("Episódio ", i+1)
        #print(observation)
        #observation = preProcessor.preprocessFrame(observation)
        passos = 0
        atravessou = 0
        #while not done:
        for passo in range(450):
            action = agent.choose_action(observation)     
            #print(action)       
            #if action == 1: passos += 1
            #elif action == 2: passos = max(0, passos - 1)
            passos += 1

            observation_, reward, done, truncated, info = env.step(action)       
            if reward == 1: 
                passos = 0     
                atravessou += 1
            #observation_ = preProcessor.preprocessFrame(observation_)
            #if reward != 1 and (passos < 0 or passos > 44): reward =- 1
            if reward != 1 and action != 1 and passo > 85: reward = -0.5
            #if action == 1: reward += 0.5
            #print("Ação | Recompensa")
            #print(action, " | ", reward)
            score += reward

            
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score) 

        avg_score = np.mean(scores[-100:])
        text = "Episode: ", i, "score %.2f" % score, "average_score % .2f" % avg_score, "epsilon %.2f" % agent.epsilon
        text2 = "\nAtravessou: ", atravessou, "\n"
        f = open("Results.txt", "a")
        f.write(str(text))
        f.write(str(text2))
        f.close()        
        print("Episode: ", i, "score %.2f" % score, "average_score % .2f" % avg_score, "epsilon %.2f" % agent.epsilon)
        
#filename = 'lunarlander_tf2.png'
#x = [i+1 for i in range(n_games)]
#plot(x, scores, eps_history, filename)