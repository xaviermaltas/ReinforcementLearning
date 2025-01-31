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

        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = list(range(self.n_outputs))

        ######

        #######################################
        ## TODO: Construcció de la xarxa neuronal
        # Xarxa comuna
        ## Construcció de la xarxa neuronal

        #xarxa comuna
        self.model_common = nn.Sequential(
            nn.Linear(self.n_inputs, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU()
        )

        # Subxarxa de la funció de Valor
        self.fc_layer_inputs = self.feature_size()


        # Subxarxa d'avantatge (A(s,a))
        self.advantage = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, self.n_outputs, bias=True)
        )
        
        # Subxarxa de valor (V(s))
        self.value = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )

        #######
        #######################################
        ## TODO: Inicialitzar l'optimitzador
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    #######################################
    ##### TODO: Funció forward #############
    def forward(self, state):
        # Connexió entre capes de la xarxa comuna
        common_out = self.model_common(state)

        # Connexió entre capes de la Subxarxa de Valor
        advantage = self.advantage(common_out)

        # Connexió entre capes de la Subxarxa d'Avantatge
        value = self.value(common_out)


        ## Agregar les dues subxarxes:
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

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
