env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',target_desire_factor=)#Modificar target_desire_factor
eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',target_desire_factor=,is_eval = True)#Modificar target_desire_factor

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats utilitzant la funció evaluate_trained_model.
#Repetir per a cada valor de target_desire_factor.