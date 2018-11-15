import numpy as np


class QLearner:

    def __init__(self, *,
                 num_states,
                 num_actions,
                 # learning_rate = alpha
                 learning_rate,
                 # discount_rate = gamma
                 discount_rate=1.0,
                 # random_action_prob = epsilon
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99,
                 dyna_iterations=0):

        self._num_states = num_states
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        self._dyna_iterations = dyna_iterations
        self._N = np.ones((num_states, num_actions))
        self.regret = 0
        self._experiences = []

        # Initialize Q to small random values.
        self._Q = np.zeros((num_states, num_actions), dtype=np.float)
        self._Q += np.random.normal(0, 0.3, self._Q.shape)


    def learn(self, initial_state, experience_func, iterations=100):
        '''Iteratively experience new states and rewards'''
        all_policies = np.zeros((self._num_states, iterations))
        all_utilities = np.zeros_like(all_policies)
        
        ### START CODE HERE ###
        all_rewards = 0
        terminal = False
        s = None
        reward = 0
        a = None
        i = 0
        while(i<iterations):
            s, a, reward, terminal, skip = self.qlearning_agent(s, a, reward, terminal, initial_state, experience_func)
            all_rewards+=reward
            if skip:
                self._random_action_prob *= self._random_action_decay_rate
                self._learning_rate *= 0.99
                all_policies[:, i] = np.argmax(self._Q,axis=1)
                all_utilities[:, i] = np.max(self._Q, axis=1)
                i+=1
            
        ### END CODE HERE ###
        print('N:',self._N)
        print(all_rewards/iterations)
        return all_policies, all_utilities
    
    def qlearning_agent(self, s, a, reward, terminal, initial_state, experience_func):
        skip = False
        if terminal:
            self._Q[s, None] = reward
            if reward ==-1:
                self.regret+=1
            return None, None, 0, False, True
        if s==None:
            # s_prime = np.random.choice(initial_state)
            s_prime = initial_state
            reward_prime = -0.1
            terminal = False
        else:
            self._N[s, a]+=1
            s_prime, reward_prime, terminal = experience_func(s, a)
            self._Q[s, a] = (1-self._learning_rate)*self._Q[s, a] + self._learning_rate*(reward + self._discount_rate*np.max(self._Q[s_prime]))
        s = s_prime
        reward = reward_prime
        a = self._exploration(s_prime,'explore')
        return s, a, reward, terminal, skip

    def _exploration(self, s, mode):
        if mode == 'e_greedy':
            prob = np.random.rand(1)
            if prob > self._random_action_prob:
                a = np.argmax(self._Q[s])
            else:
                a = np.random.choice(np.arange(self._num_actions))
        if mode=='explore':
            k = 2
            f = self._Q[s] + k / self._N[s]
            a = np.argmax(f)
        return a