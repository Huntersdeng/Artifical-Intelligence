import numpy as np
import gym
import time
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, *,
                 game,
                 # learning_rate = alpha
                 learning_rate,
                 # discount_rate = gamma
                 discount_rate=1.0,
                 # random_action_prob = epsilon
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99):
                #  dyna_iterations=0):
        self.env = gym.make(game)
        self._observation_shape = self.env.observation_space.shape
        self._num_states = 36
        self._num_actions = self.env.action_space.n
        self._action = np.arange(self._num_actions)
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        # self._dyna_iterations = dyna_iterations
        self.regret = 0
        self._experiences = []
        self._Q = np.zeros((self._num_states, self._num_actions))
        self._Q += np.random.normal(0, 0.3, self._Q.shape)
        

    # use pid control as a sample to figure out the state space, action meaning
    # and also help qlearning swiftly
    def _sample(self, observation, observation_prime):
        if (observation==0).all() and (observation_prime==0).all():
            action = np.random.choice(self._action)
        else:
            action = self._pid(observation, observation_prime)
        return action

    def _pid(self, observation, observation_prime):
        board, _, ball = self._get_position(observation)
        board_prime, _, ball_prime= self._get_position(observation_prime)
        if ball_prime[1]==0:
            ball_prime[0] = 80
        distance_prime = board_prime-ball_prime
        pid = distance_prime[0]
        #print(ball, distance_prime)
        if pid<0:
            action = 5
        else:
            action = 4
        return action

    # Q learning and epsilon greedy for exploration
    def _learner(self, state, action, reward, state_prime, reward_prime):
        self._Q[state,action] = self._Q[state,action]*(1-self._learning_rate) + self._learning_rate*(reward + self._discount_rate*np.max(self._Q[state_prime]))
        prob = np.random.rand(1)
        if prob > self._random_action_prob:
            action = np.argmax(self._Q[state_prime])
        else:
            action = np.random.choice(self._num_actions)
        return action

    # interface for learning
    # input: 
    # iterations & step
    # input_file: default None, load the data to self._Q if not None
    # output_file: save the learned Q
    def learn(self, iterations, output_file=None, input_file=None):
        if input_file!=None:
            self._Q = np.load(input_file)
        All_score = np.zeros((iterations,))
        state = None
        for i in range(iterations):
            self.env.reset()
            observation = np.zeros(self._observation_shape)
            observation_prime = np.zeros_like(observation)
            state_prime = None
            done = False
            reward_prime = 0
            while True:
                self.env.render()
                if done:
                    print("Episode finished after {} timesteps".format(i+1))
                    print('The reward after this episode is ', All_score[i])
                    self._random_action_prob *= self._random_action_decay_rate
                    self._learning_rate *= 0.9
                    break
                if (observation==0).all() and (observation_prime==0).all():
                    action = np.random.choice(self._action)
                # at the very beginning use pid samples to help learning
                elif i==0:
                    action = self._sample(observation, observation_prime)
                    _, __, ball = self._get_position(observation)
                    # if state_prime!=state:
                    _ = self._learner(state, action, reward, state_prime, reward_prime)
                else:
                    # if state_prime!=state:
                    action = self._learner(state, action, reward, state_prime, reward_prime)
                observation = observation_prime
                reward = reward_prime
                state = state_prime
                observation_prime, score, done, info = self.env.step(action)
                state_prime = self._get_move_state(observation_prime)
                a, b = np.unravel_index(state_prime, (3,12))

                # define the reward, default -0.1, -5 for lost, and 2 for nearly collide the ball
                reward_prime = -0.1
                if b == 0:
                    reward_prime = -5
                if b == 1 and a == 0:
                    reward_prime=2

                All_score[i]+=score
        if output_file!=None:
            np.save(output_file, self._Q)
        self.env.close()

    # to get the position of board, board_op and the ball from the RGB image
    def _get_position(self, observation):
        Green = np.array([92,186,92])
        Yellow = np.array([213,130,74])
        White = np.array([236,236,236])
        board_position = np.argwhere(observation[34:194]==Green)
        board = np.zeros((2,))
        board[0] = np.mean(board_position,axis=0)[0]
        board[1] = np.mean(board_position,axis=0)[1]
        ball = np.zeros((2,))
        ball_position = np.argwhere(observation[34:194]==White)
        ball[0] = np.mean(ball_position,axis=0)[0]
        ball[1] = np.mean(ball_position,axis=0)[1]
        board_op_position = np.argwhere(observation[34:194]==Yellow)
        board_op = np.zeros((2,))
        board_op[0] = np.mean(board_op_position,axis=0)[0]
        board_op[1] = np.mean(board_op_position,axis=0)[1]

        # Because sometimes there's no ball or board in the image, it might get arrays like (nan,nan)
        # To avoid that, convert the nan to zero
        board = np.nan_to_num(board)
        board_op = np.nan_to_num(board_op)
        ball = np.nan_to_num(ball)
        
        return board, board_op, ball
    
    # state space:
    # a: means the vertical distance between the board and ball, discrete values vary from 0 to 2;
    # 0 for ball in range of board's width, 1 for below the board and 2 for above the board
    # b: means the horizontal distane between the board and ball, discrete values vary from 0 to 11
    def _get_move_state(self, observation):
        board, _, ball = self._get_position(observation)
        if ball[1]<60:
            b = 11
            board[0] = 80
        elif board[1]-ball[1]<0:
            b = 0
        else:
            b = int((board[1]-ball[1])/10)+1
        if abs(board[0]-ball[0])<=6:
            a = 0
        else:
            a=(board[0]>ball[0])+1
        #print(board, board_op, ball, ball_direction)
        return np.ravel_multi_index((a,b),(3,12))
    

    # test the learned Q value
    def test(self, input_file):
        scores = 0
        self._Q = np.load(input_file)
        state = None
        print(self._Q)
        self.env.reset()
        observation = np.zeros(self._observation_shape)
        observation_prime = np.zeros_like(observation)
        state_prime = None
        done = False
        while True:
            self.env.render()
            time.sleep(0.01)
            if done:
                print('The reward after this episode is ', scores)
                self._random_action_prob *= self._random_action_decay_rate
                self._learning_rate *= 0.9
                break
            if (observation==0).all() and (observation_prime==0).all():
                action = np.random.choice(self._action)
            # at the very beginning use sample to help learning
            else:
                if state_prime!=state:
                    action = np.argmax(self._Q[state_prime])
            observation = observation_prime
            state = state_prime
            observation_prime, score, done, info = self.env.step(action)
            state_prime = self._get_move_state(observation_prime)
            scores += score
        self.env.close()

if __name__=='__main__':
    QL = QLearning(game='Pong-v0', learning_rate=0.8, discount_rate=0.9,random_action_prob=0.5, random_action_decay_rate=0.9)
    #QL.learn(50, 'Q_v7.npy')
    QL.test('Q_v7.npy')