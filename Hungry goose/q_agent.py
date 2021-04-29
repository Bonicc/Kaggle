from kaggle_environments.envs.hungry_geese.hungry_geese import *
import numpy as np
import random
from collections import deque
np.random.seed(777)

def bfs(map_, position):
    q = deque()
    distance = 0
    q.append(position)
    
    while len(q) != 0:
        distance += 1
        pqsize = len(q)
        
        for i in range(pqsize):
            position = q.popleft()
            
            for a in Action:
                npp = list(map(lambda x,y: x+y, position , a.to_row_col()))                
                r = (npp[0]+7) % 7
                c = (npp[1]+11) % 11
                
                if map_[r][c] == 2:
                    return distance+1

                if map_[r][c] != 1:
                    q.append([r,c])
                    map_[r][c] = 1
    
    return 999

class QAgent:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None
        self.food_reward = 100
        self.body_reward = -999
        self.future_head_reward = -9
        self.tail_reward = 10
        
        self.alpha = 0.3
        self.gamma = 0.99
        
    def __call__(self, observation: Observation):       
        
        cols, rows = self.configuration.columns, self.configuration.rows
                
        
        player_index = observation.index    
        
        # players head and body
        player_goose = observation.geese[player_index]
        
        # players head
        player_head = player_goose[0]
                
        # the food positions
        foods_position = observation.food

        
        # for drawing map, get position of other snakes and of mine
        # head and body position of snakes
        snakes_position = []
        for i in range(len(observation.geese)):
            if observation.geese[i] != []:
                food_near = 0
                for j in foods_position:
                    if j in adjacent_positions(observation.geese[i][0], cols, rows):
                        food_near = 1
                        
                if food_near == 1:
                    snakes_position += observation.geese[i][:]
                else:
                    snakes_position += observation.geese[i][:-1]
            

        # the position of head of future that can be candidates
        snakes_estimated_position = []
        for i in range(len(observation.geese)):  
            if observation.geese[i] !=[]:
                if observation.geese[i][0] != player_head:
                    for j in adjacent_positions(observation.geese[i][0], cols, rows):
                          snakes_estimated_position += [j]
        
        map_status = [[-0.01 for _ in range(cols)] for _ in range(rows)]
        q_table = [[{action: 0 for action in Action} for _ in range(cols)] for _ in range(rows)]
        
        for i in foods_position:
            row, col = row_col(i, cols)
            map_status[row][col] = self.food_reward

        for i in snakes_position:
            row, col = row_col(i, cols)
            map_status[row][col] = self.body_reward
        
        '''real min distance to food for each geese'''
        min_dist = []        
        for i in range(len(observation.geese)):
            if observation.geese[i] != []:
                min_dist.append(self.real_min_distance_with_bfs(map_status, observation.geese[i][0], foods_position))
            else:
                min_dist.append([999,999])
                
                
        if min_dist[player_index][0] != min(min_dist, key = lambda x: x[0]) and \
        min_dist[player_index][1] != min(min_dist, key = lambda x: x[1]) and \
        len(player_goose)>1:
            row, col = row_col(player_goose[-1],cols)
            map_status[row][col] = 1


        for i in snakes_estimated_position:
            row, col = row_col(i, cols)
            map_status[row][col] = self.future_head_reward
        
        #############################################################################    
        init_eps = 0.9
        for i in range(1000): # episodes                    
            eps = init_eps/(i+1)

            pp = row_col(player_head, cols)
            
            last_action = self.last_action
            
            for j in range(observation.step, 200): # maximum time in a episode
                
                sa = self.action_select(eps, q_table, pp, last_action)
                
                pnp = list(map(lambda x,y: x+y, pp, sa.to_row_col()))
                
                pnp[0] = (pnp[0]+rows) % rows             
                pnp[1] = (pnp[1]+cols) % cols   
                
                if map_status[pnp[0]][pnp[1]] == self.body_reward or\
                    map_status[pnp[0]][pnp[1]] == self.food_reward:
                    q_table[pp[0]][pp[1]][sa] = \
                        (1-self.alpha) * q_table[pp[0]][pp[1]][sa] + \
                        self.alpha*(map_status[pnp[0]][pnp[1]])
                    break

                else:             
                    q_table[pp[0]][pp[1]][sa] = \
                        (1-self.alpha) * q_table[pp[0]][pp[1]][sa] + \
                        self.alpha*(map_status[pnp[0]][pnp[1]]+ \
                        self.gamma*max(q_table[pnp[0]][pnp[1]].values()))

                    pp = pnp
                
                last_action = sa
                                    
        self.last_action = self.action_select(0, q_table, row_col(player_head, cols), self.last_action)
                        
        return self.last_action.name
        
        
        
    def action_select(self, eps, q_table, pp, last_action = None):
        temp_dict = {k:q_table[pp[0]][pp[1]][k] for k in q_table[pp[0]][pp[1]]}
        
        if last_action != None:
            del temp_dict[last_action.opposite()]
        
        return max(temp_dict.keys(), key = lambda k: temp_dict[k]) \
                            if eps < np.random.random_sample() else choice(list(temp_dict.keys()))
        
        
    def real_min_distance_with_bfs(self, map_status, player_head, food_position):
        cols, rows = self.configuration.columns, self.configuration.rows
        
        map_shape = np.array(map_status).shape
        map_ = np.zeros(map_shape)
        
        ph = row_col(player_head,cols)
        
        answer = []
                
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                if map_status[i][j] == self.body_reward:
                    map_[i][j] = 1 
        
        for i in food_position:
            fp = row_col(i,cols)
            
            temp_map = np.array(map_)
            temp_map[fp[0]][fp[1]] = 2
            
            answer.append(bfs(temp_map, ph))
                                  
        return answer
                
qchache = {}

def agent(obs,config):
    index = obs["index"]
    if index not in qchache:
        qchache[index] = QAgent(Configuration(config))
        
    return qchache[index](Observation(obs))