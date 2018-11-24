import random, re, datetime
import numpy as np
from math import *
from operator import eq


class Agent(object):
    def __init__(self, game):
        self.game = game

    def getAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vert1 advance
    def getAction(self, state):
        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            max_vert1_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[0][0] - action[1][0] == max_vert1_advance_one_step]
        else:
            max_vert1_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[1][0] - action[0][0] == max_vert1_advance_one_step]
        self.action = random.choice(max_actions)


class TeamNameMinimaxAgent(Agent):
    a = 1
    b = 0.16
    c = 0
    step = 0
    position = None
    #best_eval = float('-inf')
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)
        player = self.game.player(state)
        ### START CODE HERE ###
        start_state={(7, 3): 0, (12, 1): 0, (14, 4): 0, (13, 4): 0, (16, 1): 1, (15, 1): 0, (6, 2): 0, (8, 5): 0,
         (18, 1): 1, (10, 8): 0, (5, 5): 0, (11, 5): 0, (10, 7): 0, (7, 6): 0, (12, 6): 0, (8, 8): 0, (17, 2): 1, 
         (14, 1): 0, (13, 7): 0, (15, 4): 0, (9, 8): 0, (12, 3): 0, (3, 2): 2, (13, 2): 0, (8, 2): 0, (18, 2): 1, 
         (15, 3): 0, (9, 3): 0, (16, 2): 1, (7, 5): 0, (8, 7): 0, (4, 2): 2, (14, 2): 0, (9, 6): 0, (6, 5): 0, 
         (5, 3): 0, (11, 7): 0, (10, 5): 0, (11, 8): 0, (3, 1): 2, (9, 9): 0, (1, 1): 2, (12, 8): 0, (9, 1): 0, 
         (6, 6): 0, (11, 2): 0, (10, 6): 0, (7, 7): 0, (2, 1): 2, (13, 5): 0, (12, 5): 0, (17, 3): 1, (9, 4): 0, 
         (5, 1): 0, (15, 5): 0, (10, 3): 0, (7, 2): 0, (12, 2): 0, (3, 3): 2, (14, 5): 0, (13, 3): 0, (8, 1): 0, 
         (4, 4): 2, (16, 4): 1, (6, 3): 0, (11, 1): 0, (2, 2): 2, (8, 6): 0, (4, 1): 2, (10, 9): 0, (9, 7): 0, 
         (6, 4): 0, (5, 4): 0, (11, 4): 0, (10, 4): 0, (7, 1): 0, (12, 7): 0, (11, 9): 0, (17, 1): 1, (14, 6): 0, 
         (13, 6): 0, (10, 1): 0, (13, 1): 0, (8, 3): 0, (19, 1): 1, (15, 2): 0, (10, 10): 0, (9, 2): 0, (6, 1): 0, 
         (11, 3): 0, (7, 4): 0, (12, 4): 0, (4, 3): 2, (14, 3): 0, (9, 5): 0, (8, 4): 0, (5, 2): 0, (16, 3): 1, 
         (11, 6): 0, (10, 2): 0}
        
        #print(state[1].board_status)
        if eq(start_state,state[1].board_status)==1:
            self.step = 0
        if self.step < 3:
            #begin
            #print(state[1].board_status)
            self.action = self.startAction(state, player)
            print(self.action)
            self.step += 1
        else:
            if self.position=='end':
	            best_eval = float('-inf')
	            for action in legal_actions:
	                next_state = self.game.succ(state, action)
	                e = self.Heru(next_state, player)
	                if best_eval<e:
	                    best_eval = e
	            max_actions = []
	            print(best_eval)
	            for action in legal_actions:
	                next_state = self.game.succ(state, action)
	                e = self.Heru(next_state, player)
	                if abs(best_eval-e)<0.01:
	                    max_actions.append(action)
	            self.action = random.choice(max_actions)
            
            else:
            #minimax
            #TeamNameMinimaxAgent.best_eval=float('-inf')
                value = self.max_value(state, player, float('-inf'), float('inf'), 3)
                self.action = value[1]
        print(self.position)
        



    def max_value(self, state, player, a, b, t):
        #depth
        if t == 0:
            return self.Heru(state, player), None
        v = float('-inf')
        next_t = t - 1
        for action in self.game.actions(state):
            '''
            de = action[0][0] - action[1][0]
            if player == 1:
                next_e = e + de
            else:
                next_e = e - de
            '''
            if player==1 and (action[0][0]-action[1][0]<0):
            	continue
            if player==2 and (action[0][0]-action[1][0]>0):
            	continue
            
            #print('max action:',action)
            next_state = self.game.succ(state, action)
            value = self.min_value(next_state, player, a, b, next_t)
            if value[0] > v :
                Action = action
                v = value[0]
            if v >= b:
                return v, Action
            a = max(a, v) 
        return v, Action

    def min_value(self, state, player, a, b, t):
        #depth
        if t == 0:
            return self.Heru(state, player), None
        v = float('inf')
        next_t = t - 1
        for action in self.game.actions(state):
            '''
            de = action[0][0] - action[1][0]
            if player == 2:
                next_e = e + de
            else:
                next_e = e - de
            '''
            #print('min action:',action)
            if player==1 and (action[0][0]-action[1][0]<0):
            	continue
            if player==2 and (action[0][0]-action[1][0]>0):
            	continue
            next_state = self.game.succ(state, action)
            value = self.max_value(next_state, player, a, b, next_t)
            if value[0] < v :
                Action = action
                v = value[0]
            if v <= a:
                return v, Action
            a = min(b, v) 
        return v, Action

    def Heru (self, state, player):
        vertical = np.zeros((10,))
        vertical_op = np.zeros((10,))
        # horizontal = np.zeros_like(vertical)
        # horizontal_op = np.zeros_like(vertical_op)
        cnt = 0
        cnt_op = 0
        In = 0
        In_op = 0
        Out_op = 0
        Out = 0
        for point,val in state[1].board_status.items():
            if val == 1:
                vertical[cnt]=point[0]
                if point[0]>=16:
                	Out+=1
                if point[0]<=4:
                	In+=1
                # if point[0]<=10:
                #     #board.append((point[0],2*point[1]-point[0]))
                #     horizontal[cnt] =  2*point[1]-point[0]
                # else:
                #     #board.append((point[0],point[0]+2*point[1]-20))
                #     horizontal[cnt] =  point[0]+2*point[1]-20
                cnt+=1
            elif val == 2:
                vertical_op[cnt_op]=point[0]
                if point[0]>=16:
                	In_op+=1
                if point[0]<=4:
                	Out_op+=1
                # if point[0]<=10:
                #     #board.append((point[0],2*point[1]-point[0]))
                #     horizontal_op[cnt_op] =  2*point[1]-point[0]
                # else:
                #     #board.append((point[0],point[0]+2*point[1]-20))
                #     horizontal_op[cnt_op] =  point[0]+2*point[1]-20
                cnt_op+=1
        '''
        for p in point:
            if player==2:
                distance += (p[0]-1)**2+(p[1]-1)**2
            else:
                distance += (p[0]-19)**2+(p[1]-19)**2
        '''
        if player == 1:
            ave_vert = 19 - np.sum(vertical)/10 + In - Out
            variance = np.dot((vertical-ave_vert),(vertical-ave_vert).T)/10
            variance = sqrt(variance)
            # ave_hori = np.sum(horizontal)/10
            # variance_hori = np.dot((horizontal-ave_hori),(horizontal-ave_hori).T)/10
            # variance_hori = sqrt(variance_hori)
            ave_vert_op = np.sum(vertical_op)/10 + In_op - Out_op
            # eval = self.a*(ave_vert-ave_vert_op)-self.b*variance_hori-self.c*variance
            eval = self.a*(ave_vert-ave_vert_op)-self.b*variance
        else:
            ave_vert = 19 - np.sum(vertical)/10 + In - Out
            #variance = sqrt(np.dot((vertical-ave_vert),(vertical-ave_vert).T)/10)
            #ave_hori = np.sum(horizontal)/10
            #variance_hori = np.dot((horizontal-ave_hori),(horizontal-ave_hori).T)/10
            ave_vert_op = np.sum(vertical_op)/10 + In_op - Out_op
            variance_op = np.dot((vertical_op-ave_vert_op),(vertical_op-ave_vert_op).T)/10
            variance_op = sqrt(variance_op)
            # ave_hori_op = np.sum(horizontal_op)/10
            # variance_hori_op = np.dot((horizontal_op-ave_hori_op),(horizontal_op-ave_hori_op).T)/10
            # variance_hori_op = sqrt(variance_hori_op)
            # eval = self.a*(ave_vert_op-ave_vert)- self.b*variance_hori_op-self.c*variance_op
            eval = self.a*(ave_vert_op-ave_vert)- self.b*variance_op
        if ave_vert<=10:
            self.position='start'
        else:
            if ave_vert>10 and ave_vert<15:
                self.position='half'
            else:
                self.position='end' 
        return eval

    def startAction(self, state, player):
        #pattens people usually follow at the beginning of the game
        if player == 1:
            if self.step==0:
                action = ((16,1),(15,2))
            elif self.step==1:
                action = ((18,1),(14,3))
            elif self.step==2:
                action = ((19,1),(13,3))
        else:
            if self.step==0:
                action = ((4,1),(5,2))
            elif self.step==1:
                action = ((2,1),(6,3))
            elif self.step==2:
                action = ((1,1),(7,3))
        return action
        ### END CODE HERE ###
    


