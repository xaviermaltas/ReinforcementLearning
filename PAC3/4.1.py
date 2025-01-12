#eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = False,target_desire_factor=1,is_eval = True)

import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

#Define the evaluation environment
eval_env = CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', increased_actions=False, target_desire_factor=1, is_eval=True)

#Define the ranges for hyperparameters
gamma_range = (0.9, 0.999)
max_grad_norm_range = (0.3, 5.0)
n_steps_range = (8, 32)
learning_rate_range = (1e-5, 1e-1)
ent_coef_range = (1e-8, 1e-3)


#Number of trials
N_TRIALS = 100

#Function to sample hyperparameters
def sample_hyperparameters():
    gamma = random.uniform(*gamma_range)
    max_grad_norm = random.uniform(*max_grad_norm_range)
    n_steps = random.randint(*n_steps_range)
    learning_rate = random.uniform(*learning_rate_range)
    ent_coef = random.uniform(*ent_coef_range)
    return gamma, max_grad_norm, n_steps, learning_rate, ent_coef

#Perform random search
best_mean_reward = -float('inf')
best_hyperparams = None

for trial in range(N_TRIALS):
    gamma, max_grad_norm, n_steps, learning_rate, ent_coef = sample_hyperparameters()
    
    #Create the training environment
    train_env = make_vec_env(lambda: CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', increased_actions=False, target_desire_factor=1), n_envs=1)
    
    #Create the PPO model with the sampled hyperparameters
    model = PPO('MlpPolicy', train_env, gamma=gamma, max_grad_norm=max_grad_norm, n_steps=n_steps, learning_rate=learning_rate, ent_coef=ent_coef, verbose=0)
    
    #Train the model
    model.learn(total_timesteps=10_000)
    
    #Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES)
    
    #Update the best hyperparameters if the current ones are better
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        best_hyperparams = (gamma, max_grad_norm, n_steps, learning_rate, ent_coef)

#Print the best hyperparameters
print('Best hyperparameters:', best_hyperparams)