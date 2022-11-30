# policy iteration algorithm
import numpy as np
import random
import matplotlib.pyplot as plt
import time


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

    # if [state[0], state[1]] == end: reward = 100
    if done: reward = 100 #100
    else: reward = -de + 100/(ds1**2 + 1) + 100/(ds2**1 + 1)
    # else: reward = -de + np.min([2,20/(ds1**2 + 1)]) + np.min([2,20/(ds2**1 + 1)])

    return reward


def prob():
    #sigma = 1
    #p = (1/(sigma*((2*np.pi)**0.5)))*np.exp(-0.5*((pi/sigma)**2))
    n = len(A)
    return 1/n


# simulate pi within policy iteration to calculate reward values
def sim_pi():
    state1 = start
    t, rval = 0, 0

    while t < max_time_steps:
        theta_index = (np.where(theta_space == state1[2])[0]).item()
        pi = policy[state1[0],state1[1],theta_index]
        next_state, done = get_state(state1, pi)
        reward = get_reward(next_state,done)
        rval += reward
        t += 1
        state1 = next_state
        if done: break

    return rval



def policy_iteration():
    print('policy iteration')
    iterations = 0

    for iter in range(max_iter):
        rval = sim_pi()
        rvec.append(rval)
        while True: # evaluation
            print('evaluation')
            delta = 0
            for i in range(n):
                for j in range(m):
                    for k in range(o):
                        old_value = values[i,j,k]
                        new_value = 0
                        for a in A:
                            nx, done = get_state([i,j,theta_space[k]],a)
                            theta_index = (np.where(theta_space == nx[2])[0]).item()
                            reward = get_reward(nx,done)
                            new_value += prob()*(reward + gamma*values[nx[0],nx[1],theta_index])
                        values[i,j,k] = new_value
                        delta = max(delta, abs(old_value - new_value))
            print('delta: ',delta)
            iterations += 1
            if delta < error: break
        
        policy_stable = True # improvement
        print('improvement')
        for i in range(n):
            for j in range(m):
                for k in range(o):
                    old_action = policy[i,j,k]
                    new_action, max_val = None, -1*float('inf')
                    for a in A:
                        val = 0
                        for new_a in A:
                            nx, done = get_state([i,j,theta_space[k]],a)
                            theta_index = (np.where(theta_space == nx[2])[0]).item()
                            reward = get_reward(nx,done)
                            val += prob()*(reward + gamma*values[nx[0],nx[1],theta_index])
                        if val > max_val: max_val, new_action = val, a
                    policy[i,j,k] = new_action
                    if old_action != new_action: policy_stable = False
        
        iterations += 1
        print(iterations)
        if policy_stable == True: break
        
    return values, policy, iterations


# choose next action (deterministic)
def choose_deterministic_action(state):

    # find the max value in the action space for the current state
    max_q = -1*float('inf')
    greedy_actions = []
    for action in A:
        next_state, done = get_state(state, action)
        next_theta_index = (np.where(theta_space == next_state[2])[0]).item()
        q_val = values[next_state[0],next_state[1],next_theta_index]
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





# simulate trajectories
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
        action = choose_deterministic_action(state)
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
    ax.set_title(r'$\gamma = 0.9$')
    # plt.savefig('pi_g0.9.png', dpi=1200, bbox_inches='tight')
    plt.show()











# policy iteration
gamma = 0.5
min_steer, max_steer = -np.pi/6, np.pi/6
A = np.linspace(min_steer, max_steer, 7).tolist()

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

# initialize values and policy
values = np.zeros((n,m,o))
policy = np.zeros((n,m,o))
error = 1E0
max_iter = 10 # for policy iteration
max_time_steps = 50 # for simulation


rvec = []
t1 = time.perf_counter()
values, policy, iterations = policy_iteration()
t2 = time.perf_counter()
print('time: ', t2 - t1)
print('iter: ',iterations)
# print(np.array2string(values,separator=','))
# print(np.array2string(policy,separator=','))
plt.plot(rvec,color='k')
print(rvec)
plt.xlabel('policy improvement iterations')
plt.ylabel('reward')
plt.savefig('pi_reward.png', dpi=1200, bbox_inches='tight')
plt.show()

simulate()
