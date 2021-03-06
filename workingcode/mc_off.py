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
    numEvalEpisodes = 1000
    rewards = np.zeros(numEvalEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test target policy')
    for i in range(numEvalEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = targetPolicy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1

    wins /= numEvalEpisodes
    losses /= numEvalEpisodes
    draws /= numEvalEpisodes
    
    end_time = time.monotonic()

    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    print('reward average', np.mean(rewards))
    print(timedelta(seconds=end_time - start_time))
    
    plt.plot(rewards)
    plt.show()
