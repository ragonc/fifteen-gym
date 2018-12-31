import gym
import gym_settemezzo
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

start_time = time.monotonic()

if __name__ == '__main__':
    env = gym.make('Settemezzo-v0')
    EPS = 0.1
    GAMMA = 0.9

    agentSumSpace = [i for i in range(1, 30)]
    dealerShowCardSpace = [i + 1 for i in range(15)]
    actionSpace = [0, 1]  # stick or hit
    stateSpace = []

    Q = {}
    C = {}
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for action in actionSpace:
                Q[((total, card), action)] = 0
                C[((total, card), action)] = 0
            stateSpace.append((total, card))

    targetPolicy = {}
    for state in stateSpace:
        values = np.array([Q[(state, a)] for a in actionSpace])
        best = np.random.choice(np.where(values == values.max())[0])
        targetPolicy[state] = actionSpace[best]

    numEpisodes = 1000000
    for i in range(numEpisodes):
        memory = []
        if i % 100000 == 0:
            print('starting episode', i)
        behaviorPolicy = {}
        for state in stateSpace:
            rand = np.random.random()
            if rand < 1 - EPS:
                behaviorPolicy[state] = [targetPolicy[state]]
            else:
                behaviorPolicy[state] = actionSpace
        observation = env.reset()
        epRewards = 0
        done = False
        
        while not done:
            action = np.random.choice(behaviorPolicy[observation])
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], action, reward))

        G = 0
        W = 1
        last = True
        for playerSum, dealerCard, action, reward in reversed(memory):
            sa = ((playerSum, dealerCard), action)
            if last:
                last = False
            else:
                C[sa] += W
                Q[sa] += (W / C[sa]) * (G - Q[sa])
                values = np.array([Q[(state, a)] for a in actionSpace])
                best = np.random.choice(np.where(values == values.max())[0])
                targetPolicy[state] = actionSpace[best]
                if action != targetPolicy[state]:
                    break
                if len(behaviorPolicy[state]) == 1:
                    prob = 1 - EPS
                else:
                    prob = EPS / len(behaviorPolicy[state])
                W *= 1 / prob
            G = GAMMA * G + reward
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
    np.save("dq_last_rewards", totalRewards)
    plt.plot(segments)
    plt.show()
