class DQNAgent:
    def __init__(self, env, main_network, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_episodes=300, device=None):
        ######################################
        ## TODO 1: Declarar variables
        self.env = None  # TODO: assignar l'entorn
        self.main_network = None  # TODO: assignar la xarxa principal
        self.target_network = None  # xarxa objectiu (còpia de la principal)
        self.buffer = None  # TODO: assignar el buffer de repetició d'experiències
        self.epsilon = None  # TODO: assignar el valor inicial de epsilon
        self.eps_decay = None  # TODO: assignar la velocitat de decaïment de epsilon
        self.batch_size = None  # TODO: assignar la mida del batch
        self.nblock = None  # bloc dels X últims episodis per calcular la mitjana de recompenses
        self.initialize()
        self.min_episodes = None  # TODO: assignar el nombre mínim d'episodis
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Configurar el dispositiu (CPU o GPU)

    def initialize(self):
        ######################################
        ## TODO 3: Inicialitzar el necessari
        pass

    ## Prendre una nova acció
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # acció aleatòria durant el burn-in o en la fase d'exploració (epsilon)
            action = None
        else:
            # acció basada en el valor de Q (elecció de l'acció amb millor Q)
            action = None
            self.step_count += 1
        # TODO: prendre un 'step', obtenir un nou estat i recompensa. Desar l'experiència al buffer

        # TODO: reiniciar l'entorn si s'ha completat l'episodi ('if done')

    ## Entrenament
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000, REWARD_THRESHOLD=9000):

        self.gamma = gamma

        # Omplir el buffer amb N experiències aleatòries (burn-in)
        print("Omplint el buffer de repetició d'experiències...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Entrenament...")
        while training:
            self.state0 = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            while not gamedone:
                # L'agent pren una acció
                gamedone = self.take_step(self.epsilon, mode='train')
                ##################################################################################
                ##### TODO 4: Actualitzar la xarxa principal segons la freqüència establerta #######

                ########################################################################################
                ### TODO 6: Sincronitzar la xarxa principal i la xarxa objectiu segons la freqüència establerta #####

                if gamedone:
                    episode += 1
                    #######################################################################################
                    ### TODO 7: calcular la mitjana de recompenses dels últims X episodis i emmagatzemar #####

                    self.update_loss = []

                    #######################################################################################
                    ### TODO 8: Comprovar que encara queden episodis. Parar l'aprenentatge si s'arriba al límit

                    print("\rEpisodi {:d} Recompenses Mitjanes {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprovar si s'ha arribat al límit d'episodis
                    if episode >= max_episodes:
                        training = False
                        print('\nLímit d\'episodis assolit.')
                        print('\nEntorn resolt en {} episodis!'.format(episode))
                        break

                    #######################################################################################
                    ### TODO 9: Afegir el mínim d'episodis requerits
                    if mean_rewards >= REWARD_THRESHOLD and self.min_episodes < episode:
                        training = False
                        print('\nLímit d\'episodis assolit.')
                        print('\nEntorn resolt en {} episodis!'.format(episode))
                        break

                    #################################################################################
                    ###### TODO 9: Actualitzar epsilon segons la velocitat de decaïment fixada ########

    ## Càlcul de la pèrdua
    def calculate_loss(self, batch):
        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(self.device)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals).to(self.device)
        # Obtenim els valors de Q objectiu. El paràmetre detach() evita que aquests valors actualitzin la xarxa objectiu
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach().to(self.device)
        qvals_next[dones_t.bool()] = 0

        #################################################################################
        ### TODO: Calcular equació de Bellman
        expected_qvals = None

        #################################################################################
        ### TODO: Calcular la pèrdua (MSE)
        loss = None
        return loss

    def update(self):
        self.main_network.optimizer.zero_grad()  # eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.batch_size)  # seleccionem un conjunt del buffer
        loss = self.calculate_loss(batch)  # calculem la pèrdua
        loss.backward()  # fem la diferència per obtenir els gradients
        self.main_network.optimizer.step()  # apliquem els gradients a la xarxa neuronal
        # Guardem els valors de pèrdua
        self.update_loss.append(loss.detach().cpu().numpy())
