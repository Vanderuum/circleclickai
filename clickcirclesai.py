import pygame
import gym
from gym import spaces
import math
import time
import random
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
from stable_baselines3 import PPO

#Built in environment check
from stable_baselines3.common.env_checker import check_env

class click_circle(gym.Env):

    metadata = {'render.modes': ['human']}
    #Direction constants
    n_actions = 9 #9 possible steps each turn
    MOVE_NORTH = 0
    MOVE_NORTHEAST = 1
    MOVE_EAST = 2
    MOVE_SOUTHEAST = 3
    MOVE_SOUTH = 4
    MOVE_SOUTHWEST = 5
    MOVE_WEST = 6
    MOVE_NORTHWEST = 7
    CLICKMOUSE = 8
    #Grid label constants
    EMPTY = 0
    CIRCLE = 1
    WALL=2
    #REWARD_PER_STEP = 0 # reward for every step taken, gets into infinite loops if >0
    #Define Max steps to avoid infinite loops
    #REWARD_WALL_HIT = -20 #should be lower than -REWARD_PER_STEP_TOWARDS_FOOD to avoid hitting wall intentionally
    REWARD_PER_STEP_TOWARDS_CIRCLE = 0.5 #give reward for moving towards food and penalty for moving away
    REWARD_PER_CIRCLE = 2000 
    MAX_STEPS_AFTER_CIRCLE = 2000 #stop if we go too long without food to avoid infinite loops
            

    def __init__(self):
        super(click_circle, self).__init__()
        self.resetarray = np.array([120, 120, 0], dtype=np.int32)
        self.circlex = self.resetarray[0]
        self.circley = self.resetarray[1]
        pygame.init()
        self.window = pygame.display.set_mode((640,480))
        pygame.display.set_caption("Click the Circles Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.move=True
        self.font = pygame.font.Font(None, 36)
         #score defaults
        self.score = self.resetarray[2]
        self.score_increment = 1
        self.move = True
        self.resetarraycopy = self.resetarray.copy()
        self.stepnum = 0; self.last_circle_step=0
        self.moveCommandX = 1
        self.moveCommandY = -1
        mousePos = pygame.mouse.get_pos()
        self.mousetargetx = mousePos[0]
        self.mousetargety = mousePos[1]

        # The action space
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # The observation space, "position" is the coordinates of the head; "direction" is which way the sanke is heading, "grid" contains the full grid info
        self.observation_space = gym.spaces.Dict(
            spaces={
                "positioncursor": gym.spaces.Box(low=np.array([0,0]), high=np.array([640,480]), shape=(2,), dtype=np.int32),
                "positioncircle": gym.spaces.Box(low=np.array([0,0]), high=np.array([640,480]), shape=(2,), dtype=np.int32),
            })

    def reset(self):
        self.stepnum = 0; self.last_circle_step=0
        pygame.mouse.set_pos(320,240)
        # Reset to initial positions
        self.resetarray = self.resetarraycopy.copy()
        self.circlex = self.resetarray[0]
        self.circley = self.resetarray[1]
        self.score = self.resetarray[2]
        self.moveCommandX = 1
        self.moveCommandY = -1
        return self._get_obs()
    
    def _get_obs(self):
        mousePos = pygame.mouse.get_pos()
        self.mousetargetx = mousePos[0]
        self.mousetargety = mousePos[1]
        self.cursorposition = np.array([self.mousetargetx, self.mousetargety], dtype=np.int32)
        self.circleposition = np.array([self.circlex, self.circley], dtype=np.int32)
            #return observation in the format of self.observation_space

        return {"positioncursor": self.cursorposition,
                "positioncircle": self.circleposition}   

    def step(self, action):
        if keyboard.is_pressed('x'):
           pygame.QUIT()
        
        self.circlex += self.moveCommandX
        self.circley += self.moveCommandY

           # mouse position
        
        mousePos = pygame.mouse.get_pos()
        self.mousetargetx = mousePos[0]
        self.mousetargety = mousePos[1]
        

         #calculate the 8 positions around the mouse cursor
        self.cursorEAST = (self.mousetargetx + 10, self.mousetargety)
        self.cursorEASTlist = [self.mousetargetx + 10, self.mousetargety]
        if self.cursorEASTlist[0] > 620:
            self.cursorEAST = (600, self.cursorEASTlist[1])
        if self.cursorEASTlist[1] > 460:
            self.cursorEAST = (self.cursorEASTlist[0], 440)
        if self.cursorEASTlist[0] < 20:
            self.cursorEAST = (40, self.cursorEASTlist[1])
        if self.cursorEASTlist[1] < 20:
            self.cursorEAST = (self.cursorEASTlist[0], 40)
        self.cursorSOUTH = (self.mousetargetx, self.mousetargety + 10)
        self.cursorSOUTHlist = [self.mousetargetx, self.mousetargety + 10]
        if self.cursorSOUTHlist[0] > 620:
            self.cursorSOUTH = (600, self.cursorSOUTHlist[1])
        if self.cursorSOUTHlist[1] > 460:
            self.cursorSOUTH = (self.cursorSOUTHlist[0], 440)
        if self.cursorSOUTHlist[0] < 20:
            self.cursorSOUTH = (40, self.cursorSOUTHlist[1])
        if self.cursorSOUTHlist[1] < 20:
            self.cursorSOUTH = (self.cursorSOUTHlist[0], 40)
        self.cursorSOUTHEAST = (self.mousetargetx + 10, self.mousetargety + 10)
        self.cursorSOUTHEASTlist = [self.mousetargetx + 10, self.mousetargety + 10]
        if self.cursorSOUTHEASTlist[0] > 620:
            self.cursorSOUTHEAST = (600, self.cursorSOUTHEASTlist[1])
        if self.cursorSOUTHEASTlist[1] > 460:
            self.cursorSOUTHEAST = (self.cursorSOUTHEASTlist[0], 440)
        if self.cursorSOUTHEASTlist[0] < 20:
            self.cursorSOUTHEAST = (40, self.cursorSOUTHEASTlist[1])
        if self.cursorSOUTHEASTlist[1] < 20:
            self.cursorSOUTHEAST = (self.cursorSOUTHEASTlist[0], 40)
        self.cursorSOUTHWEST = (self.mousetargetx - 10, self.mousetargety + 10)
        self.cursorSOUTHWESTlist = [self.mousetargetx - 10, self.mousetargety + 10]
        if self.cursorSOUTHWESTlist[0] > 620:
            self.cursorSOUTHWEST = (600, self.cursorSOUTHWESTlist[1])
        if self.cursorSOUTHWESTlist[1] > 460:
            self.cursorSOUTHWEST = (self.cursorSOUTHWESTlist[0], 440)
        if self.cursorSOUTHWESTlist[0] < 20:
            self.cursorSOUTHWEST = (40, self.cursorSOUTHWESTlist[1])
        if self.cursorSOUTHWESTlist[1] < 20:
            self.cursorSOUTHWEST = (self.cursorSOUTHWESTlist[0], 40)
        self.cursorWEST = (self.mousetargetx - 10, self.mousetargety)
        self.cursorWESTlist = [self.mousetargetx - 10, self.mousetargety]
        if self.cursorWESTlist[0] > 620:
            self.cursorWEST = (600, self.cursorWESTlist[1])
        if self.cursorWESTlist[1] > 460:
            self.cursorWEST = (self.cursorWESTlist[0], 440)
        if self.cursorWESTlist[0] < 20:
            self.cursorWEST = (40, self.cursorWESTlist[1])
        if self.cursorWESTlist[1] < 20:
            self.cursorWEST = (self.cursorWESTlist[0], 40)
        self.cursorNORTHWEST = (self.mousetargetx - 10, self.mousetargety - 10)
        self.cursorNORTHWESTlist = [self.mousetargetx - 10, self.mousetargety - 10]
        if self.cursorNORTHWESTlist[0] > 620:
            self.cursorNORTHWEST = (600, self.cursorNORTHWESTlist[1])
        if self.cursorNORTHWESTlist[1] > 460:
            self.cursorNORTHWEST = (self.cursorNORTHWESTlist[0], 440)
        if self.cursorNORTHWESTlist[0] < 20:
            self.cursorNORTHWEST = (40, self.cursorNORTHWESTlist[1])
        if self.cursorNORTHWESTlist[1] < 20:
            self.cursorNORTHWEST = (self.cursorNORTHWESTlist[0], 40)
        self.cursorNORTH = (self.mousetargetx, self.mousetargety - 10)
        self.cursorNORTHlist = [self.mousetargetx, self.mousetargety - 10]
        if self.cursorNORTHlist[0] > 620:
            self.cursorNORTH = (600, self.cursorNORTHlist[1])
        if self.cursorNORTHlist[1] > 460:
            self.cursorNORTH = (self.cursorNORTHlist[0], 440)
        if self.cursorNORTHlist[0] < 20:
            self.cursorNORTH = (40, self.cursorNORTHlist[1])
        if self.cursorNORTHlist[1] < 20:
            self.cursorNORTH = (self.cursorNORTHlist[0], 40)
        self.cursorNORTHEAST = (self.mousetargetx + 10, self.mousetargety - 10)
        self.cursorNORTHEASTlist = [self.mousetargetx + 10, self.mousetargety - 10]
        if self.cursorNORTHEASTlist[0] > 620:
            self.cursorNORTHEAST = (600, self.cursorNORTHEASTlist[1])
        if self.cursorNORTHEASTlist[1] > 460:
            self.cursorNORTHEAST = (self.cursorNORTHEASTlist[0], 440)
        if self.cursorNORTHEASTlist[0] < 20:
            self.cursorNORTHEAST = (40, self.cursorNORTHEASTlist[1])
        if self.cursorNORTHEASTlist[1] < 20:
            self.cursorNORTHEAST = (self.cursorNORTHEASTlist[0], 40)
        

        #mouse vs self collision detection
        self.distx = self.mousetargetx - self.circlex
        self.disty = self.mousetargety - self.circley
        self.distance = math.sqrt((self.distx*self.distx)+(self.disty*self.disty))
        
        # boundary collision detection and self movement
    
        if self.circlex > 630:
            self.moveCommandX *= -1
            self.circlex = 630
        if self.circley > 470:
            self.moveCommandY *= -1
            self.circley = 470
        if self.circlex < 10:
            self.moveCommandX *= -1
            self.circlex = 10
        if self.circley < 10:
            self.moveCommandY *= -1
            self.circley = 10

        mousePos = pygame.mouse.get_pos()
        self.mousetargetx = mousePos[0]
        self.mousetargety = mousePos[1]

        self.distbeforex = self.mousetargetx - self.circlex
        self.distbeforey = self.mousetargety - self.circley
        self.distancebefore = math.sqrt((self.distbeforex*self.distbeforex)+(self.distbeforey*self.distbeforey))
        #Get direction for cursor/click
        if action == self.MOVE_NORTH:
            pygame.mouse.set_pos(self.cursorNORTH) #move cursor north
            self.stepnum += 1
        elif action == self.MOVE_NORTHEAST:
            pygame.mouse.set_pos(self.cursorNORTHEAST) #move cursor northeast
            self.stepnum += 1        
        elif action == self.MOVE_EAST:
            pygame.mouse.set_pos(self.cursorEAST) #move cursor east
            self.stepnum += 1
        elif action == self.MOVE_SOUTHEAST:
            pygame.mouse.set_pos(self.cursorSOUTHEAST) #move cursor southeast
            self.stepnum += 1
        elif action == self.MOVE_SOUTH:
            pygame.mouse.set_pos(self.cursorSOUTH) #move cursor south
            self.stepnum += 1
        elif action == self.MOVE_SOUTHWEST:
            pygame.mouse.set_pos(self.cursorSOUTHWEST) #move cursor southwest
            self.stepnum += 1
        elif action == self.MOVE_WEST:
            pygame.mouse.set_pos(self.cursorWEST) #move cursor west
            self.stepnum += 1
        elif action == self.MOVE_NORTHWEST:
            pygame.mouse.set_pos(self.cursorNORTHWEST) #move cursor northwest
            self.stepnum += 1
        elif action == self.CLICKMOUSE:
            if self.distance<=10: # check for distance between self and mouse
                if random.randint(0,1) == 1: # random self movement after respawn
                    self.moveCommandX *= 1.1
                else:
                    self.moveCommandX *= -1.1
                if random.randint(0,1) == 1:
                    self.moveCommandY *= 1.1
                else:
                    self.moveCommandY *= -1.1
                self.circlex = random.randint(0,640) #random self placement
                self.circley = random.randint(0,480)
                self.score += self.score_increment #increase score
        else:
            raise ValueError("Action=%d is not part of the action space"%(action))

        mousePos = pygame.mouse.get_pos()
        self.mousetargetx = mousePos[0]
        self.mousetargety = mousePos[1]
        
        self.distafterx = self.mousetargetx - self.circlex
        self.distaftery = self.mousetargety - self.circley
        self.distanceafter = math.sqrt((self.distafterx*self.distafterx)+(self.distaftery*self.distaftery))
        
        #Check what is at the new position
        
        done = False; reward = 0 #by default the game goes on and no reward   
        if action == self.CLICKMOUSE:
            if self.distance<=10:
                reward += self.REWARD_PER_CIRCLE
                self.stepnum = 0
        #if self.mousetargetx > 630:
         #   reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail
        #if self.mousetargety > 470:
        #    reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail
        #if self.mousetargetx < 10:
        #    reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail
        #if self.mousetargety < 10:
        #    reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail
#             else:
#                 reward += self.REWARD_PER_STEP
                
        #Update distance to food and reward if closer
        
        if self.distancebefore > self.distanceafter:
            reward += self.REWARD_PER_STEP_TOWARDS_CIRCLE #reward for getting closer to food
        elif self.distancebefore < self.distanceafter:
            reward -= self.REWARD_PER_STEP_TOWARDS_CIRCLE #penalty for getting further
        
        #Stop if we played too long without clicking circle
        if ( (self.stepnum - self.last_circle_step) > self.MAX_STEPS_AFTER_CIRCLE ): 
            done = True    
        self.stepnum += 1

        self.window.fill((0,0,0))
        x = self.circlex
        y = self.circley
        pygame.draw.circle(self.window,(0,0,255),(x,y),10)
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 0))
        self.window.blit(score_text, (10, 10))
        pygame.display.update()

        return  self._get_obs(), reward, done, {}
    
    def render(self):
        pass

    def close(self):
        pass

env = click_circle()
# If the environment doesn't follow the interface, an error will be thrown
check_env(env, warn=True)
#
##Logging
#log_dir = "log"
#os.makedirs(log_dir, exist_ok=True)
## wrap it
#env = Monitor(env, log_dir)
#
#
##Callback, this built-in function will periodically evaluate the model and save the best version
#eval_callback = EvalCallback(env, best_model_save_path='./log/',
#                             log_path='./log/', eval_freq=5000,
#                             deterministic=False, render=False)
#

##Train the agent
#max_total_step_num = 2000
#
def learning_rate_schedule(progress_remaining):
    start_rate = 0.0001 #0.0003
    #Can do more complicated ones like below
    #stepnum = max_total_step_num*(1-progress_remaining)
    #return 0.003 * np.piecewise(stepnum, [stepnum>=0, stepnum>4e4, stepnum>2e5, stepnum>3e5], [1.0,0.5,0.25,0.125 ])
    return start_rate * progress_remaining #linearly decreasing

#PPO_model_args = {
#    "learning_rate": learning_rate_schedule, #decreasing learning rate #0.0003 #can be set to constant
#    "gamma": 0.99, #0.99, discount factor for futurer rewards, between 0 (only immediate reward matters) and 1 (future reward equivalent to immediate), 
#    "verbose": 1, #change to 1 to get more info on training steps
#    #"seed": 137, #fixing the random seed
#    "ent_coef": 0.0, #0, entropy coefficient, to encourage exploration
#    "clip_range": 0.2 #0.2, very roughly: probability of an action can not change by more than a factor 1+clip_range
#}
#starttime = time.time()
#model = PPO('MultiInputPolicy', env,**PPO_model_args)
##Load previous best model parameters, we start from that
#if os.path.exists("log/best_model.zip"):
#    model.set_parameters("log/best_model.zip")
#model.learn(max_total_step_num,callback=eval_callback)
#dt = time.time()-starttime
#print("Calculation took %g hr %g min %g s"%(dt//3600, (dt//60)%60, dt%60) )

#from stable_baselines3.common import results_plotter
# Helper from the library, a bit hard to read but immediately useable
#results_plotter.plot_results(["log"], 1e7, results_plotter.X_TIMESTEPS,'')
#plt.savefig("circle_rewards.png",dpi=150, bbox_inches="tight")
#plt.show()

#Load back the best model
model = PPO('MultiInputPolicy', env)
model.set_parameters("log/best_model.zip")
#from stable_baselines3.common.evaluation import evaluate_policy
## Evaluate the trained model
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
#print("Best model's reward: %3.3g +/- %3.3g"%(mean_reward,std_reward))

# Test the trained agent and save animation
obs = env.reset()

n_steps = 8000
tot_reward = 0
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    tot_reward += reward
    time.sleep(0.01)
    print("Step {}".format(step + 1),"Action: ", action, 'Tot. Reward: %g'%(tot_reward))
    #print('position=', obs['position'], 'direction=', obs['direction'])
    #env.render(mode='console')
    #frames.append([ax.imshow(env.render(mode='rgb_array'), animated=True)])
    if done:
        print("Game over!", "tot. reward=", tot_reward)
        break
#fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) #to remove white bounding box 
#anim = animation.ArtistAnimation(fig, frames, interval=int(1000/fps), blit=True,repeat_delay=1000)
#anim.save("snake_best.gif",dpi=150)
