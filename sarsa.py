# SARSA algorithm
import numpy as np
import random
import matplotlib.pyplot as plt
import time




# choose next action (epsilon-greedy)
def choose_action(state):

    # find the max Q value in the action space for the current state
    max_q = -1*float('inf')
    greedy_actions = []
    for action in A:
        next_state, done = get_state(state, action)
        next_theta_index = (np.where(theta_space == next_state[2])[0]).item()
        q_val = Q[next_state[0],next_state[1],next_theta_index]
        #if q_val > max_q:  
        #    max_q, greedy_action = q_val, action
        
        if q_val > max_q:  
            max_q = q_val
            greedy_actions = [] # remove other actions from the list
            greedy_actions.append(action)
        elif q_val == max_q:
            max_q = q_val
            greedy_actions.append(action)
        
    if np.random.uniform(0, 1) > epsilon:
        a = random.choice(A) # exploration
    else:
        #a = greedy_action # exploitation
        a = random.choice(greedy_actions)
    return a



# choose next action (deterministic)
def choose_deterministic_action(state):

    # find the max Q value in the action space for the current state
    max_q = -1*float('inf')
    greedy_actions = []
    for action in A:
        next_state, done = get_state(state, action)
        next_theta_index = (np.where(theta_space == next_state[2])[0]).item()
        q_val = Q[next_state[0],next_state[1],next_theta_index]
        #if q_val > max_q:  
        #    max_q, greedy_action = q_val, action
        
        if q_val > max_q:  
            max_q = q_val
            greedy_actions = [] # remove other actions from the list
            greedy_actions.append(action)
        elif q_val == max_q:
            max_q = q_val
            greedy_actions.append(action)
        
    # a = greedy_action
    a = random.choice(greedy_actions)
    return a


# round value to closest value in given array
def round(val, arr):
    index = (np.abs(val - arr)).argmin()

    return arr[index]


# get the next state
def get_state(state, action):
    x, y, theta = state[0], state[1], state[2]

    # rounding takes care of boundaries and strange angles
    next_theta = round(theta + action, theta_space) # rounded to the nearest angle in theta-space
    next_x = round(x + dt*v*np.cos(theta), x_space) # rounded to nearest integer in x-space
    next_y = round(y + dt*v*np.sin(theta), y_space) # rounded to nearest integer in y-space

    next_state = [next_x, next_y, next_theta]

    done = False
    # if [next_x, next_y] == end: done = True
    if next_x > end[0] - b and next_x < end[0] + b and next_y > end[1] - b and next_y < end[1] + b: done=True

    return next_state, done



def get_reward(state,done):

    de = ((end[0] - state[0])**2 + (end[1] - state[1])**2)**0.5 # Euclidean distance to end state
    ds1 = ((source_1[0] - state[0])**2 + (source_1[1] - state[1])**2)**0.5 # Euclidean distance to power source 1
    ds2 = ((source_2[0] - state[0])**2 + (source_2[1] - state[1])**2)**0.5 # Euclidean distance to power source 2

    if done: reward = 1
    # else: reward = -de + np.min([2,10/(ds1**2)]) + np.min([2,10/(ds2**1)])
    else: reward = -de + 100/(ds1**2 + 1) + 100/(ds2**1 + 1)

    return reward



#Function to learn the Q-value
def update(state, next_state, reward):
    theta_index = (np.where(theta_space == state[2])[0]).item()
    next_theta_index = (np.where(theta_space == next_state[2])[0]).item()

    term = reward + gamma*Q[next_state[0], next_state[1], next_theta_index] - Q[state[0], state[1], theta_index]
    Q[state[0], state[1], theta_index] = Q[state[0], state[1], theta_index] + alpha*term


def sarsa():
    ep_num = 0
    for episode in range(total_episodes):
        print('episode: ',ep_num)
        r = 0 # cumulative episode reward sum
        state = start
        action = choose_action(state)

        # while True:
        t = 0
        while t < max_time_steps:
            next_state, done = get_state(state, action)
            reward = get_reward(next_state,done)
            r = r + reward
            next_action = choose_action(next_state)

            # update the Q value
            update(state, next_state, reward)

            # update current state and action
            state = next_state
            action = next_action

            # if state is terminal, break
            if done: break

            t += 1

        ep_num += 1
        rvec[episode] = 1*r



# simulate epsilon-greedy or deterministic trajectories
def simulate():
    img = np.ones((n,m,3))
    img[start[0], start[1], :] = [0,1,0] # start is green

    fig, ax = plt.subplots()
    ax.scatter(source_1[0],source_1[1],color='blueviolet',s=100)
    ax.scatter(source_2[0],source_2[1],color='blueviolet',s=100)
    # ax.legend(['power source'])

    state = start
    t = 0
    while t < max_time_steps:
        action = choose_deterministic_action(state) # deterministic
        # action = choose_action(state) # stochastic
        next_state, done = get_state(state, action)

        ax.plot([state[1], next_state[1]], [state[0], next_state[0]], color='k', linewidth=2)

        state = next_state

        img[state[0], state[1], :] = [0,0,0] # path is black

        t += 1
        if done: break

    img[end[0], end[1], :] = [1,0,0] # end is red
    
    ax.imshow(img)
    ax.invert_yaxis()
    ax.set_xlabel('x-displacement (m)')
    ax.set_ylabel('y-displacement (m)')
    ax.set_title(r'$\gamma = 0.8$, $\alpha = 0.5$, $\epsilon = 0.9$, $k=1000$')
    # plt.savefig('sarsa_g0.8_loop_2.png', dpi=1200, bbox_inches='tight')
    plt.show()



# run SARSA learning:
total_episodes = 10000
alpha = 0.5
gamma = 0.9 # 0.5
epsilon = 0.9
max_time_steps = 60

min_steer, max_steer = -np.pi/6, np.pi/6
A = np.linspace(min_steer, max_steer, 7)

n = 60 # x discretizations
m = 30 # y discretizations
o = 20 # heading angle discretizations
dt = 1 # timestep
v = 4 # velocity
b = 3 # done radius

x_space = np.arange(n)
y_space = np.arange(m)
theta_space = np.linspace(-np.pi, np.pi, o)

end = [55,15]
start = [1,15,theta_space[9]]
source_1, source_2 = [5,15], [25,35]

# initialize Q matrix
Q = np.zeros((n,m,o))
# initialize cumulative reward vector
rvec = np.zeros((total_episodes))

sarsa()

plt.plot(rvec,color='k')
plt.title(r'$\alpha=0.5$, $\gamma=0.9$, $\epsilon=0.9$')
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('sarsa_reward.png', dpi=1200, bbox_inches='tight')
plt.show()


simulate()