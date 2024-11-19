######################## SOLUCIÓ ###########################
import numpy as np
import gymnasium as gym
from collections import defaultdict

#SARSA Algorithm
def sarsaAlgorithm(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min):
    #Init Qtalbe
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #store deltas
    deltas = []

    #Generate an episode following an epsilon-soft policy
    def generate_episode(Q, epsilon):
        episode = []
        state, _ = env.reset()
        
        # Choose an initial action following the epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample() #Random action
        else:
            action = np.argmax(Q[state]) #Greedy action -> max Q value
        
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            #store the episode (state, action, reward)
            episode.append((state, action, reward))
            
            #Update state and action
            state = next_state
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample() # Random action
            else:
                next_action = np.argmax(Q[state]) #greedy action -> max Q value
            
            action = next_action #update the action
            
            done = terminated or truncated
        
        return episode
    
    #loop per each episode
    for episode_num in range(1, num_episodes + 1):
        #new episode
        episode = generate_episode(Q, epsilon)
        
        #update qtable for each step in the episode
        for t in range(len(episode) - 1):
            state, action, reward = episode[t]
            next_state, next_action, _ = episode[t + 1]
            
            #store qvalues before compute delta
            old_q_value = Q[state][action]
            
            #Update Qvaluw using SARSA rules
            Q[state][action] += learning_rate * (reward + discount_factor * Q[next_state][next_action] - old_q_value)
            
            #Compute TD error and append
            td_error = abs(Q[state][action] - old_q_value)
            deltas.append(td_error)
        
        #update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        #Print progressx
        if episode_num % 1000 == 0:
            print(f"Episode {episode_num}/{num_episodes} completed.")

    #optimal policy based on Qtable
    optimal_policy = {state: np.argmax(actions) for state, actions in Q.items()}

    print("\nOptimal policy based on SARSA:")
    for state, action in optimal_policy.items():
        print(f"State {state}: Action {action}")

    return Q, deltas

# Parameters
num_episodes = 10000
learning_rate = 0.2
gamma = 1.0  #discount factor
initial_epsilon = 0.5
epsilon_decay = 0.9
epsilon_min = 0.05

#Init epsilon
epsilon = initial_epsilon

# Run the SARSA algorithm
Q_sarsa, deltas = sarsaAlgorithm(env, num_episodes, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min)

# Qsarsa and deltas print (uncomment to see)
# print("\nQ-values from SARSA:")
# for state, actions in Q_sarsa.items():
#     print(f"State {state}: {actions}")

######################## SOLUCIÓ ###########################
import matplotlib.pyplot as plt

#Moving average 100 episodes 
moving_average = np.convolve(deltas, np.ones(100)/100, mode='valid')

#TD error evolution plot
plt.figure(figsize=(10, 6))
plt.plot(deltas, label='TD Error per Step', color='blue')
plt.plot(moving_average, label='Moving Average (100 episodes)', color='red', linestyle='dashed')
plt.xlabel('Episodes')
plt.ylabel('TD Error')
plt.title('TD Error Evolution and Moving Average (SARSA)')
plt.legend()
plt.show()

######################## SOLUCIÓ ###########################
print_policy(Q_sarsa, width=4, height=4)

######################## SOLUCIÓ ###########################
execute_episode(Q_sarsa,env)