import numpy as np
import gym
import time
import matplotlib.pyplot as plt
# from ql_learn_gym import *

env = gym.make('Pong-v0')
# QL = QLearning(state_size=(160,160),
#  			   num_actions = env.action_space.n,
# 			   learning_rate=0.8)
for i_episode in range(1):
	env.reset()
	# state = np.zeros((210,160,3))
	# pre_state = np.zeros_like(state)
	for t in range(1000):
		env.render()
		action = env.action_space.sample()
		# pre_state = state
		state, reward, done, info = env.step(action)
		time.sleep(0.1)
		print(reward, done, info)
		# print(reward, done, info)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break