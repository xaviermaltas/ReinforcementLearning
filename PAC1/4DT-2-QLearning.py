######################## SOLUCIÓ ###########################
import numpy as np
import gymnasium as gym
from collections import defaultdict

# Q-learning Algorithm
def qLearning(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min):
    #init Qtable
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #Store deltas
    deltas = []

    #Loop though episodes
    for episode_num in range(1, num_episodes + 1):
        state, _ = env.reset()  #reset env
        done = False
        episode_td_error = [] #current episode td error

        #run episode
        while not done:
            #Select action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample() #Random action
            else:
                action = np.argmax(Q[state]) #Greedy action 
            
            #Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            #Updata Qvalue using qLearning rule
            old_q_value = Q[state][action] #Save qvalue to compute delta
            best_next_action = np.argmax(Q[next_state]) #Take best action for the next state
            Q[state][action] += learning_rate * (reward + discount_factor * Q[next_state][best_next_action] - old_q_value)
            
            #Calculate and store the TD error
            td_error = abs(Q[state][action] - old_q_value)
            episode_td_error.append(td_error)
            
            #update current state
            state = next_state
            
            #Check if episode completed
            done = terminated or truncated
        
        #Append max TD error for this episode
        deltas.append(np.max(episode_td_error))
        
        #update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        #Print progress
        if episode_num % 1000 == 0:
            print(f"Episode {episode_num}/{num_episodes} completed.")

    #optimal policy based on Qtable
    print("\nOptimal policy based on Q-learning:")
    optimal_policy = {state: np.argmax(actions) for state, actions in Q.items()}
    for state, action in optimal_policy.items():
        print(f"State {state}: Action {action}")
    
    return Q, deltas


# Parameters
num_episodes = 5000
learning_rate = 0.4
gamma = 1.0  #discount factor
initial_epsilon = 0.5
epsilon_decay = 0.9
epsilon_min = 0.05

#init epsilon
epsilon = initial_epsilon


Q_qlearning, deltas = qLearning(env, num_episodes, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min)



######################## SOLUCIÓ ###########################
import matplotlib.pyplot as plt

#Moving average 100 episodes 
window_size = 100
moving_avg_td_errors = np.convolve(deltas, np.ones(window_size)/window_size, mode='valid')

#TD error evolution plot and moving average
plt.figure(figsize=(10, 6))
plt.plot(deltas, label='Max TD Error per Episode')
plt.plot(np.arange(window_size - 1, num_episodes), moving_avg_td_errors, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('TD Error')
plt.title('TD Error Evolution in Q-learning')
plt.legend()
plt.show()

######################## SOLUCIÓ ###########################
print_policy(Q_qlearning, width=4, height=4)

######################## SOLUCIÓ ###########################
execute_episode(Q_qlearning,env)