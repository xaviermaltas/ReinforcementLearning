class NeuralNetStockMarket(torch.nn.Module):

    ###################################
    ### Inicialització i model ###
    def __init__(self, env, learning_rate=1e-3, optimizer=None, device=None):
        """
        Paràmetres
        ==========
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """
        ######################################
        ##Inicialitzar paràmetres
        super(NeuralNetStockMarket, self).__init__()
        self.n_inputs = env.observation_space.shape[0]  #assignar la mida de l'espai d'estats
        self.n_outputs = env.action_space.n  #assignar la mida de l'espai d'accions
        self.actions = list(range(self.n_outputs))  #assignar l'array d'accions possibles
        self.learning_rate = learning_rate  #assignar el valor de la taxa d'aprenentatge

        # Definir el dispositiu (CPU o GPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #######################################
        ## Construcció de la xarxa neuronal
        #construir el model Seqüencial
        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, 256, bias=True), #primera capa - 256 neurones
            nn.ReLU(),
            nn.Linear(256, 128, bias=True), #segona capa - 128 neurones
            nn.ReLU(),
            nn.Linear(128, 64, bias=True), #tercera capa - 64 neurones
            nn.ReLU(),
            nn.Linear(64, self.n_outputs, bias=True) #capa de sortida
        ).to(self.device)

        #######################################
        ## Inicialitzar l'optimitzador
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

    ### Mètode e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            #random action (exploració)
            action = np.random.choice(self.actions)
        else:
            #action based on qvalues
            qvals = self.get_qvals(state)
            action = torch.max(qvals, dim=-1)[1].item()
        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        return self.model(state_t)