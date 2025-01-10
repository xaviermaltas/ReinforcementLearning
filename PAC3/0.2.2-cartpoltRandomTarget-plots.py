import numpy as np
import matplotlib.pylab as plt
X_THRESHOLD = env.unwrapped.x_threshold*1.25 #Limit de posicio del carro
ANGLE_THRESHOLD = env.unwrapped.theta_threshold_radians*1.25 #Limit de l'angle del pal
def analyze_results(observations, rewards):
    for step_idx, step in enumerate(observations):
        #Extract positions, angles and actions
        positions = [obs[0][0] for obs in step] #relative position to the target
        angles = [obs[0][2] for obs in step] #pole angles
        actions = [obs[1] for obs in step] #actions

        #Scatter plot
        plt.figure(figsize=(10, 6))
        for pos, angle, action in zip(positions, angles, actions):
            marker = 'x' if action == 1 else 'o'  #X right | O left
            plt.scatter(pos, angle, c='blue', marker=marker, alpha=0.7)

        #Highlight start and end points
        plt.scatter(positions[0], angles[0], c='green', label='Start', s=100) #start piont
        plt.scatter(positions[-1], angles[-1], c='red', label='End', s=100) #end point

        # Customize plot
        plt.xlabel('Position relative to target')
        plt.ylabel('Pole angle (radians)')
        plt.title(f'Distance-Angle Trajectory (Step {step_idx + 1})')
        plt.ylim(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
        plt.xlim(-X_THRESHOLD, X_THRESHOLD)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    #Reward evolution plot
    cumulative_rewards = np.cumsum(rewards) #cumulative reward
    
    #lines
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, marker='o', linestyle='-', color='orange', alpha=0.8, label='Step Reward')
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, marker='s', linestyle='--', color='blue', alpha=0.8, label='Cumulative Reward')
    #annotations
    plt.text(len(rewards) - 1, rewards[-1] + 0.1, 'Step Reward', color='orange', fontsize=10)
    plt.text(len(cumulative_rewards) - 1, cumulative_rewards[-1] + 0.1, 'Cumulative Reward', color='blue', fontsize=10)
    
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Evolution')
    plt.grid(alpha=0.3)
    plt.show()

    #print steps
    total_steps = sum(len(step) for step in observations)
    print(f'Number of steps taken: {total_steps}')

# TODO: passar inputs necessaris
analyze_results(observations, rewards)