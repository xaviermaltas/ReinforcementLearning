#TODO: Completar codi
env = CartPoleEnvRandomTarget(render_mode='human',reward_function = 'default')

#Init variables
observations = []
current_step = [] #temporary storage
rewards = []
actions = []
steps = 0

#Reset env
state, _ = env.reset()
done = False

while not done:
    #random action
    action = env.action_space.sample() 
    actions.append(action) #save action

    #Info before step
    print(f"Step: {steps}, Action: {action}, State: {state}")

    #execute action
    next_state, reward, terminated, truncated, _ = env.step(action)

    #Status after step
    print(f"Next State: {next_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}\n")

    current_step.append((state,action)) #save temporary state-action pair
    rewards.append(reward)#save reward
    
    #Check episode end
    if terminated or truncated:
        print("Episode ended.")
        done = True
        observations.append(current_step)
        break
        
    steps += 1 #increase steps
    state = next_state #next state to current state for next step

#Close env
env.close()

#Print steps
print(f"Number of steps taken: {steps}")

# # Print the values of observations and rewards
# print("\nObservations:")
# for episode in observations:
#     for step in episode:
#         print(step)

# print("\nRewards:")
# print(rewards)

# Convertim les observacions a un format numpy per facilitar l'anàlisi
observations = np.array(observations, dtype=object) 
rewards = np.array(rewards)