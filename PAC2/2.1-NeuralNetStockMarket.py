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
        ## TODO: Inicialitzar paràmetres
        super(NeuralNetStockMarket, self).__init__()
        self.n_inputs = None  # TODO: assignar la mida de l'espai d'estats
        self.n_outputs = None  # TODO: assignar la mida de l'espai d'accions
        self.actions = None  # TODO: assignar l'array d'accions possibles
        self.learning_rate = None  # TODO: assignar el valor de la taxa d'aprenentatge

        # Definir el dispositiu (CPU o GPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #######################################
        ## Construcció de la xarxa neuronal
        self.model = None  # TODO: construir el model Seqüencial aquí

        #######################################
        ## Inicialitzar l'optimitzador
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

    ### Mètode e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = None  # TODO: acció aleatòria
        else:
            qvals = None  # TODO: calcular el valor de Q per a l'estat
            action = torch.max(qvals, dim=-1)[1].item()
        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        return self.model(state_t)