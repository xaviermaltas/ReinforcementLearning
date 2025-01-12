target_desire_factor = 1
reward_functions = ['custom1', 'custom2']  # Funcions de recompensa a comparar
STEPS = 20000
EVAL_FREQ = 2000

# Funció per entrenar l'agent per una reward_function
def train_agent(reward_function):
    # Creem l'entorn d'entrenament i d'avaluació amb el valor fix de target_desire_factor
    env = CartPoleEnvRandomTarget(render_mode=None, reward_function=reward_function, target_desire_factor=target_desire_factor)
    eval_env = CartPoleEnvRandomTarget(render_mode=None, reward_function=reward_function, target_desire_factor=target_desire_factor, is_eval=True)

    # Creem l'agent (PPO en aquest cas)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Avaluació inicial abans de l'entrenament
    print(f"Evaluació inicial per reward_function = {reward_function}:")
    initial_eval = evaluate_trained_model(model, eval_env)  # Avaluem el model sense entrenar
    print(f"Initial evaluation: {initial_eval}")

    # Callback per a l'avaluació durant l'entrenament
    eval_callback = EvalCallback(
        DummyVecEnv([lambda: eval_env]),  # Creem l'entorn vectoritzat dins del callback
        best_model_save_path=f"./best_model_{reward_function}", 
        log_path=f"./logs_{reward_function}", 
        eval_freq=EVAL_FREQ,  # Establert a EVAL_FREQ
        verbose=1
    )

    # Entrenem el model
    print(f"Iniciant l'entrenament per reward_function = {reward_function}...")
    model.learn(total_timesteps=STEPS, callback=eval_callback)  # Establert a STEPS

    # Carreguem el millor model
    model = PPO.load(f"./best_model_{reward_function}/best_model.zip")
    
    return model, eval_env

# Funció per avaluar el model entrenat
def evaluate_trained_model(model, eval_env):
    total_reward = 0
    total_steps = 0
    for _ in range(10):  # Evaluem durant 10 episodis
        obs, _ = eval_env.reset()  # Ens assegurem d'obeir només l'observació
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs)
            result = eval_env.step(action)
            # Desempaquetem els primers 4 elements
            obs, reward, done, _ = result[:4]
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
        total_steps += steps
    
    average_reward = total_reward / 10
    average_steps = total_steps / 10
    return average_reward, average_steps

# Entrenament dels dos models amb funcions de recompensa diferents
models = []
eval_envs = []

for reward_function in reward_functions:
    print(f"\n{'-'*50}")
    print(f"Entrenament amb reward_function = {reward_function}")
    model, eval_env = train_agent(reward_function)
    models.append(model)
    eval_envs.append(eval_env)
    print(f"\n{'-'*50}")

# Avaluació comparativa dels dos models
for reward_function, (model, eval_env) in zip(reward_functions, zip(models, eval_envs)):
    print(f"\nEvaluant el millor model amb reward_function = {reward_function}:")
    average_reward, average_steps = evaluate_trained_model(model, eval_env)
    print(f"Recompensa mitjana: {average_reward}, Passos mitjans per episodi: {average_steps}")
