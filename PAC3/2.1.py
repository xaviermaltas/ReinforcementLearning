# env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',target_desire_factor=)#Modificar target_desire_factor
# eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',target_desire_factor=,is_eval = True)#Modificar target_desire_factor

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats utilitzant la funció evaluate_trained_model.
#Repetir per a cada valor de target_desire_factor.

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats utilitzant la funció evaluate_trained_model.
#Repetir per a cada valor de target_desire_factor.

target_desire_factors = [0, 0.5, 1]
STEPS = 20000
EVAL_FREQ = 2000

# Funció per entrenar l'agent per un valor de target_desire_factor
def train_agent(target_desire_factor):
    # Creem l'entorn d'entrenament i avaluació amb el valor actual de target_desire_factor
    env = CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', target_desire_factor=target_desire_factor)
    eval_env = CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', target_desire_factor=target_desire_factor, is_eval=True)

    # Creem l'agent (PPO en aquest cas)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Avaluació inicial abans de l'entrenament
    print(f"Evaluació inicial per target_desire_factor = {target_desire_factor}:")
    initial_eval = evaluate_trained_model(model, eval_env)  # Avaluem el model sense entrenar
    print(f"Initial evaluation: {initial_eval}")

    # Iniciem l'entrenament
    print(f"Iniciant l'entrenament per target_desire_factor = {target_desire_factor}...")
    
    # Callback per a l'avaluació durant l'entrenament
    eval_callback = EvalCallback(
        DummyVecEnv([lambda: eval_env]),  # Creem l'entorn vectoritzat dins del callback
        best_model_save_path=f"./best_model_{target_desire_factor}", 
        log_path=f"./logs_{target_desire_factor}", 
        eval_freq=EVAL_FREQ,  # Establert a EVAL_FREQ
        verbose=1
    )

    # Entrenem el model
    model.learn(total_timesteps=STEPS, callback=eval_callback)  # Establert a STEPS

    # Carreguem el millor model
    model = PPO.load(f"./best_model_{target_desire_factor}/best_model.zip")
    
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

# Funció per avaluar els tres models
def evaluate_all_models(models, eval_envs):
    for target_desire_factor, (model, eval_env) in zip(target_desire_factors, zip(models, eval_envs)):
        print(f"\nAvaluant el millor model per target_desire_factor = {target_desire_factor}:")
        average_reward, average_steps = evaluate_trained_model(model, eval_env)
        print(f"Recompensa mitjana: {average_reward}, Passos mitjans per episodi: {average_steps}")

# Entrenament dels tres models
models = []
eval_envs = []
for target_desire_factor in target_desire_factors:
    print(f"\n{'-'*50}")
    print(f"Entrenament amb target_desire_factor = {target_desire_factor}")
    model, eval_env = train_agent(target_desire_factor)
    models.append(model)
    eval_envs.append(eval_env)
    print(f"\n{'-'*50}")

# Avaluem els tres models
evaluate_all_models(models, eval_envs)
