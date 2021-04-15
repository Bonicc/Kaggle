from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import numpy as np
np.random.seed(777)

def agent(obs_dict, config_dict):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    alpha = 0.3
    gamma = 0.95
        
    def action_select(eps, q_table,player_position, action_length):
        action_index = np.random.randint(action_length)
        if eps< np.random.random_sample():
            action_index = np.argmax(q_table[player_position[0]][player_position[1]])
        return action_index
        
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    
    # 내 얼굴과 몸통 (player index에 따라 해당 goose의 위치 계산 가능)    
    player_goose = observation.geese[player_index]
    
    
    # 0번째는 내 얼굴
    player_head = player_goose[0]
    
    # 얼굴의 위치 (int형)을 주어진 맵의 위치로 변환(configuration.columns 는 맵의 컬럼개수)
    player_row, player_column = row_col(player_head, configuration.columns) #    
    
    # 첫번쨰 음식과 두번째 음식이 있음.
    foods_position = observation.food
    
    # for drawing map, get position of other snakes and of mine
    snakes_position = []
    for i in range(len(observation.geese)):
        snakes_position += observation.geese[i]        
        
    map_status = [[-0.01 for _ in range(configuration.columns)] for _ in range(configuration.rows)]
    q_table = [[[0 for _ in range(len(Action))] for _ in range(configuration.columns)] for _ in range(configuration.rows)]
    
    for i in snakes_position:
        row, col = row_col(i, configuration.columns)
        map_status[row][col] = -99
        
    for i in foods_position:
        row, col = row_col(i, configuration.columns)
        map_status[row][col] = 100
    
    action_name = [Action.EAST.name, Action.WEST.name, Action.SOUTH.name, Action.NORTH.name]        
    action_set = [[0,1],[0,-1],[1,0],[-1,0]]
    
    #for i in range(7):
    #    print(map_status[i])
    
    init_eps = 0.9
    for i in range(1000):
        eps = init_eps/(i+1)
        
        pp = [player_row,player_column]
        
        for j in range(5000):      
            ai = action_select(eps,q_table,pp,len(Action))
            pnp = list(map(lambda x,y: x+y, pp,action_set[ai]))
            pnp[0] = (pnp[0]+7)%7
            pnp[1] = (pnp[1]+11)%11                        
                        
            if map_status[pnp[0]][pnp[1]] == -99 or\
            map_status[pnp[0]][pnp[1]] == 100:
                q_table[pp[0]][pp[1]][ai] = \
                    (1-alpha) * q_table[pp[0]][pp[1]][ai] + \
                    alpha*(map_status[pnp[0]][pnp[1]])
                break
                
            else:
                q_table[pp[0]][pp[1]][ai] = \
                    (1-alpha) * q_table[pp[0]][pp[1]][ai] + \
                    alpha*(map_status[pnp[0]][pnp[1]]+
                    gamma*max((q_table[pnp[0]][pnp[1]])))
                
                pp = pnp
            
    #print(action_name[np.argmax(q_table[player_row][player_column])])
    #print(q_table[player_row][player_column])
    return action_name[np.argmax(q_table[player_row][player_column])]
            