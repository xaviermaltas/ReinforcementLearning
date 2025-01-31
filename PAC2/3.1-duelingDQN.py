import torch.autograd as autograd

class duelingDQN(torch.nn.Module):

    def __init__(self, env, device=None, learning_rate=1e-3):

        """
        Paràmetres
        ==========
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """

        ###################################
        #### TODO: Inicialitzar variables ####
        super(duelingDQN, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_inputs = #TODO
        self.n_outputs = #TODO
        self.actions = #TODO

        ######

        #######################################
        ## TODO: Construcció de la xarxa neuronal
        # Xarxa comuna
        ## Construcció de la xarxa neuronal

        self.model_common = #TODO

        # Subxarxa de la funció de Valor
        self.fc_layer_inputs = self.feature_size()


        self.advantage  = #TODO

        # Recordeu adaptar-les a CPU o GPU

        # Subxarxa de l'Avantatge A(s,a)
        self.value = #TODO

        #######
        #######################################
        ## TODO: Inicialitzar l'optimitzador
        self.optimizer = #TODO


    #######################################
    ##### TODO: Funció forward #############
    def forward(self, state):
        # Connexió entre capes de la xarxa comuna
        common_out = #TODO

        # Connexió entre capes de la Subxarxa de Valor
        advantage = #TODO

        # Connexió entre capes de la Subxarxa d'Avantatge
        value = #TODO


        ## Agregar les dues subxarxes:
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = #TODO

        return action
    #######



    ### Mètode e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action = torch.max(qvals, dim=-1)[1].item()
        return action


    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        return self.forward(state_t)

    def feature_size(self):
        dummy_input = torch.zeros(1, *env.observation_space.shape).to(self.device)
        return self.model_common(autograd.Variable(dummy_input)).view(1, -1).size(1)
