######################## SOLUCIÓ COMPLETA ###########################
import numpy as np
import gymnasium as gym
from collections import defaultdict

# Parameters
num_episodes = 50000
initial_epsilon = 0.5
epsilon_decay = 0.999
epsilon_min = 0.05
gamma = 1.0  # Discount factor

# Initialize the environment
env = gym.make("Gym-Gridworlds/Ex2-4x4-v0", render_mode=None)

# Initialize epsilon
epsilon = initial_epsilon

# Initialize Q-table (action-value function)
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Initialize returns for state-action pairs
returns_sum = defaultdict(float)
returns_count = defaultdict(float)

# Function to generate an episode following epsilon-soft policy
def generate_episode(env, Q, epsilon):
    episode = []
    state, _ = env.reset()
    done = False
    
    while not done:
        # epsilon-soft policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    
    return episode

# On-policy first-visit MC control algorithm for epsilon-soft policies
for episode_num in range(1, num_episodes + 1):
    # Generate a new episode
    episode = generate_episode(env, Q, epsilon)
    
    # Initialize variables to compute cumulative reward
    seen_state_action_pairs = set()
    G = 0  # Cumulative reward initialized to 0
    
    # Process the episode in reverse order to calculate G
    for state, action, reward in reversed(episode):
        G = reward + gamma * G  # Update cumulative reward
        
        # First-visit check for state-action pair
        if (state, action) not in seen_state_action_pairs:
            seen_state_action_pairs.add((state, action))
            
            # Update returns and Q-value for the state-action pair
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
    
    # Update epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    # Print progress every 1000 episodes
    if episode_num % 1000 == 0:
        print(f"Episode {episode_num}/{num_episodes} completed.")

# Extract the optimal policy from the Q-table
optimal_policy = {state: np.argmax(actions) for state, actions in Q.items()}

print("Optimal policy learned.")
print("Q-values for each state-action pair:")
for state, actions in Q.items():
    print(f"State {state}: {actions}")

print("Optimal policy (ordered by states):")
for state in sorted(optimal_policy.keys()):
    print(f"State {state}: Action {optimal_policy[state]}")

# print("Optimal policy:")
# for state, action in optimal_policy.items():
#     print(f"State {state}: Action {action}")

######################## SOLUCIÓ ###########################
Q_mc = Q #Computed Q-table from MC algorithm

#Q-table print
print("---------Q-table----------")
for state, actions in Q_mc.items():
    print(f"State {state}:")
    for action, q_value in enumerate(actions):
        print(f"  Action {action}: Q-value = {q_value:.2f}")
print("\n----------------------------")

# Optimal policy sorted by states
optimal_policy = {state: np.argmax(actions) for state, actions in Q_mc.items()}
print("\nOptimal policy based on Q_mc (ordered by states)\n")
for state in sorted(optimal_policy.keys()):
    print(f"State {state}: Action {optimal_policy[state]}")

######################## SOLUCIÓ ###########################
# Action mapping dictionary
switch_action = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
    4: "Stay",
}

def print_policy(Q, width, height):
    print("Optimal Policy based on Q-values\n")
    
    # Iterate over each cell of the grid
    for row in range(height):
        row_policy = []
        for col in range(width):
            state = row * width + col  # Coordinates (row, col) to a linear index
            
            # Check if the state exists in the Q-table
            if state in Q:
                best_action = np.argmax(Q[state])  # Take action with highest Q value
                row_policy.append(switch_action.get(best_action, "None"))  # Append action
            else:
                row_policy.append("None")  # If the state does not exist, add "None"
        
        # Print the policy for the row
        print(" ".join(row_policy))

print_policy(Q_mc, width=4, height=4)

######################## SOLUCIÓ ###########################
def execute_episode(Q, env):
    #Init state by resetting the env
    state, _ = env.reset() 
    done = False #Termination episode flag to false
    trajectory = []  # Store agent's trajectory
    total_reward = 0  # Init total reward

    while not done:
        # Get best action based on Qtable 
        best_action = np.argmax(Q[state])
        
        # take chosen action
        next_state, reward, terminated, truncated, _ = env.step(best_action)
        
        # Store state,action,reward
        trajectory.append((state, best_action, reward))
        
        # Update reward and state
        total_reward += reward
        state = next_state
        
        # Check if the episode it is finished
        done = terminated or truncated
    
    # Display agent's trajectory
    print("\nAgent's trajectory:")
    for step in trajectory:
        state, action, reward = step
        print(f"State: {state}, Action: {action}, Reward: {reward}")
    
    # Disply total reward
    print(f"\nTotal return for the episode: {total_reward}")

execute_episode(Q,env)