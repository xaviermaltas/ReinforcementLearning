STEPS = 10_000 #BAIXEM ELS STEPS A 10_000 per fer la cerca més ràpida
EVAL_EPISODES = 18
EVAL_FREQ = 2_000
N_TRIALS = 100  #Només farem 100 trials amb diferents combinacions d'hiperparàmetres

eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = False,target_desire_factor=1,is_eval = True)

#TODO: Cerca hiperparàmetres

print ('Best hiperparams:')