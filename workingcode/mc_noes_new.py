import gym
import gym_settemezzo
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

start_time = time.monotonic()

if __name__ == '__main__':
    env = gym.make('Settemezzo-v0')
    
    start_time = time.time()  # for tracking run time of method

    # model hyperparameters
    EPS = 0.09
    GAMMA = 0.3

#Setting up the Q-list that we are going to be working with
    Q = {}
    agentSumSpace = [i for i in range(1, 30)]
    dealerShowCardSpace = [i + 1 for i in range(15)]
    actionSpace = [0, 1]  # stick or hit
    
#Creation and initialisation of the variables
    stateSpace = []
    returns = {}
    pairsVisited = {}
    
#I teration over the state space and creation of tuples out of those
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for action in actionSpace:
                Q[((total, card), action)] = 0
                returns[((total, card), action)] = 0
                pairsVisited[((total, card), action)] = 0 #keep track of how many times you visited a point
            stateSpace.append((total, card)) #create a list out of it and append the results

    policy = {} #tells the agent what to do in any given space
    for state in stateSpace:
        policy[state] = np.random.choice(actionSpace) #start with stick or hit with 50% probability (random policy)

    numEpisodes = 10000000
    totalRewards = np.zeros(numEpisodes)

    for i in range(numEpisodes):
        statesActionsReturns = [] #empty list to keep track of state actions
        memory = [] #memory to append the episodes

        if i % 1000000 == 0: #to keep track of the training (not mandatory, but useful)
            print('Starting Episode', i)
        observation = env.reset()
        epRewards = 0
        done = False #reset till you are not done 

#To actually play the game you need the following cycle
        while not done:
            action = policy[observation] #select an action, given the observation of the environment
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            memory.append((observation[0], observation[1], action, reward))
            observation = observation_ #reset to the new state
        memory.append((observation[0], observation[1], action, reward)) #when the episode is over append the final state

        G = 0 
        last = True
        for playerSum, dealerCard, action, reward in reversed(memory):
            if last:
                last = False
            else:
                statesActionsReturns.append((playerSum, dealerCard, action, G)) #return after the first visit
            G = GAMMA * G + reward

        statesActionsReturns.reverse() #reverse to put in chronological order
        statesActionsVisited = []

        for playerSum, dealerCard, action, G in statesActionsReturns:
            sa = ((playerSum, dealerCard), action) #state and actions tuple
            if sa not in statesActionsVisited:
                pairsVisited[sa] += 1
                # incremental implementation
                # new estimate = 1 / N * [sample - old estimate]
                returns[(sa)] += (1 / pairsVisited[(sa)]) * (G - returns[(sa)])
                Q[sa] = returns[sa]
                rand = np.random.random()
                if rand < 1 - EPS: #use epsilon greedy to find the best action
                    state = (playerSum, dealerCard)
                    values = np.array([Q[(state, a)] for a in actionSpace])
                    best = np.random.choice(np.where(values == values.max())[0]) #instead of using argmax, a random tiebreaker is used
                    policy[state] = actionSpace[best]
                else:
                    policy[state] = np.random.choice(actionSpace)
                statesActionsVisited.append(sa)
        #we don't to get negative 
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0

        totalRewards[i] = epRewards

# Summary printouts of performance
    segments = []
    split = np.split(np.array(totalRewards), 100)
    print("Progression of segmented average scores: ")
    for sub in split:
        print(np.mean(sub))
        segments.append(np.mean(sub))
    idx = int(numEpisodes / 10)
    print("'Trained' (avg of last tenth) reward:", np.mean(totalRewards[-idx:]))
    print("Average reward:", np.mean(totalRewards))
    unique, counts = np.unique(np.array(totalRewards), return_counts=True) 
    print(dict(zip(unique, counts)))
    print("Total runtime: ", round(time.time() - start_time, 2), " seconds")
    np.save("noes_rewards", totalRewards)
    plt.plot(segments)
    plt.show()
