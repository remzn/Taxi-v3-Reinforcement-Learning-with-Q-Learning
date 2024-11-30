import gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())


action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1
gamma = 0.6
epsilon = 0.1


for i in tqdm(range(1, 1000000)):
    state, _ = env.reset()
    
    done = False
    
    while not done :
        
        if random.uniform(0, 1)  < epsilon:
            action = env.action_space.sample()
            
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, done, info, _ = env.step(action)
        
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
        
print("finish of training")

total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(episodes)):
    
    state, _ = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done :
        
        action = np.argmax(q_table[state])
            
        next_state, reward, done, info, _ = env.step(action)
        
        state = next_state
        
        if reward == -10:
            penalties += 1
            
        epochs += 1
        
    total_epoch += epochs
    total_penalties += penalties
    

print("Results :", episodes)
print("\nAverage Penalties :", total_penalties/episodes)


