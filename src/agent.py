import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.points = 0
        self.s = None
        self.a = None

        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)
        

        If self._train is True, this function should update the Q table and return an action

        If self._train is False, the agent should simply return the best action based on the Q table.
        '''

        #Discretize the state of the environment
        snake_head_x, snake_head_y = int(state[0]), int(state[1])
        food_x, food_y = int(state[3]), int(state[4])
        snake_body = state[2]
        #Boolean of to keep track of whether the head is in the same coordinates of a body part 

        adjoining_wall_x, adjoining_wall_y = 0, 0
        #if ((snake_head_x < 40 or snake_head_x > 480) and (snake_head_y < 40 or snake_head_y > 480)): #out of bounds
        #    adjoining_wall_x, adjoining_wall_y = 0, 0
        if (snake_head_x==40): adjoining_wall_x = 1
        if (snake_head_x==480): adjoining_wall_x = 2
        if (snake_head_y==40): adjoining_wall_y = 1
        if (snake_head_y==480): adjoining_wall_y = 2

        food_dir_x, food_dir_y = 0, 0
        if (food_x < snake_head_x): food_dir_x = 1
        if (food_x > snake_head_x): food_dir_x = 2
        if (food_y < snake_head_y): food_dir_y = 1
        if (food_y > snake_head_y): food_dir_y = 2

        adjoining_body_left, adjoining_body_right, adjoining_body_bottom, adjoining_body_top = 0, 0, 0, 0
        for segment in snake_body:
            if ((segment[0]+40)==snake_head_x and snake_head_y==segment[1]): adjoining_body_left = 1
            if ((segment[0]-40)==snake_head_x and snake_head_y==segment[1]): adjoining_body_right = 1
            if ((segment[1]+40)==snake_head_y and snake_head_x==segment[0]): adjoining_body_bottom = 1
            if ((segment[1]-40)==snake_head_y and snake_head_x==segment[0]): adjoining_body_top = 1

        if ((points - self.points) > 0):
            reward = 1
        elif (dead):
            reward = -1
        else:
            reward = -0.1

        self.points = points
        
        #current state tuple
        new_state = tuple([adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right])
        #~~~~~~~~~~~TRAINING MODE: Steps~~~~~~~~~~
        #1. Update Q table
        #2. Get next action using exploration policy
        #3. Update N-table with that action 
        
        
        if self._train:
            if self.s is not None and self.a is not None:
                #Perform TD Update on Q-table if initial state is not None
                alpha = self.C / (self.C + self.N[self.s][self.a])     
                self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha*(reward + self.gamma*np.max(self.Q[new_state]) - self.Q[self.s][self.a])

            #Dead State: Update Q-table and reset the game
            if dead:
                self.reset()

            else:
                #update to new state
                self.s = new_state
                #select next action based on exploration policy , 0=up, 1=down, 2=left, 3=right
                action_scores = []
                index, best_action = 0, -10000000
                for i in range(0,4):
                    if (self.N[self.s][i] < self.Ne):
                        action_scores.append(1)
                    else:
                        action_scores.append(self.Q[self.s][i])
                    
                #if all scores are equal, prioritize right > left > down > up
                if len(set(action_scores))==1:
                    self.a = 3
                else:
                    for ac in action_scores:
                        if (ac>=best_action):
                            best_action = ac
                            self.a = index
                        index += 1
                    
                #update N-table
                self.N[self.s][self.a] += 1
                            
        #TESTING MODE: Simply return best action from Q-table        
        else:
            max_value = max(self.Q[new_state])
            self.a = (list(self.Q[new_state])).index(max_value)

        self.actions[0] = self.a
        return self.actions[0]
