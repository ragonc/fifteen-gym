
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_settemezzo
import time


def maxAction(Q1, Q2, state):
    #values = np.array([Q1[state, a] + Q2[state, a] for a in range(2)])
    #print(values)
    hit = Q1[state, 1] + Q2[state, 1]   
    stick = Q1[state, 0] + Q2[state, 0]
    # print('state number: ' + str(state))
    # print('hit: ' + str(hit))
    # print('stick: ' + str(stick))
    # action = np.argmax(values)
    return int(hit > stick)


# takes the raw player sum value and dealer show card value
# and returns the indices of the matching state
def getState(observation):
    #dealerShowSpace = [0.5, 1, 2, 3, 4, 5, 6, 7]  # for us this will be 0 (1) to 14
    # playerSumSpace = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #                   12, 13, 14]  # for us this will be 0.5 - 14 or 1 - 28 
    playerSumSpace = list(range(1, 30))  # up to 7 + 7 (28) however, will still need to learn to stay on 15/7.5 :)
    dealerShowSpace = [1, 2, 4, 6, 8, 10, 12, 14]  # for us this will be 0 (1) to 14
    player, dealer = observation
    state = (playerSumSpace.index(player) * 8) + dealerShowSpace.index(dealer)
    # print('obs: ' + str(player) + ' ' + str(dealer))
    # print('calculated state: ' + str(state))
    # player = playerSumSpace.index(player)
    # dealer = dealerShowSpace.index(dealer)

    return state  # NOTE: should return index of player * dealer.size + dealer


if __name__ == '__main__':
    env = gym.make('Settemezzo-v0')

    start_time = time.time()  # for tracking run time of method

    # model hyperparameters
    ALPHA = 0.0001  # learning rate
    GAMMA = 0.3  # discount factor
    EPS = 1.0  # starting exploration rate
    epsMIN = 0.005  # minimum exploration rate
    epsRF = 0.999  # per-episode exploration reduction factor

    EPISODES = 10000000
    total_reports = 10  # how often to report episode#. Default 10 times per run
    reporting_factor = EPISODES / total_reports
    totalRewards = np.zeros(EPISODES)

    # player possibilities: 14; dealer poss: 8
    Q1 = np.zeros((30 * 8, 2))
    Q2 = np.zeros((30 * 8, 2))

    for i in range(EPISODES):
        if i % reporting_factor == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            s = getState(observation)
            rand = np.random.random()
            methodCalled = False
            if rand < (1 - EPS):
                a = maxAction(Q1, Q2, s)
                methodCalled = True
            else:
                # a = env.action_space.sample()  # NOTE: this should be just 1 or 0, but it's not sampling correctly (sometimes)
                a = np.random.randint(0, 2)
                methodCalled = False
            # print(a)
            if a > 1:
                print('erroneous action choice: ' + str(methodCalled))
                a = 1
            observation_, reward, done, info = env.step(a)
            epRewards += reward
            # print('observation: ' + str(observation_[0]) + ' ' + str(observation_[1]))
            s_ = getState(observation_)
            rand = np.random.random()
            if rand <= 0.5:
                a_ = maxAction(Q1, Q1, s)
                Q1[s, a] = Q1[s, a] + ALPHA * (reward + GAMMA * Q2[s_, a_] - Q1[s, a])
            elif rand > 0.5:
                a_ = maxAction(Q2, Q2, s)
                Q2[s, a] = Q2[s, a] + ALPHA * (reward + GAMMA * Q1[s_, a_] - Q2[s, a])
            observation = observation_

        # Defining an exploration reduction function, so as the episode # increases the change of exploration
        # decreases
        if EPS > epsMIN:
            EPS = EPS * epsRF
        # print(EPS)
        # print(epRewards)
        totalRewards[i] = epRewards

    # print(states)
    # print(Q1)
    # print(Q2)

    # Summary printouts of performance
    segments = []
    split = np.split(np.array(totalRewards), 100)
    print("Progression of segmented average scores: ")
    for sub in split:
        print(np.mean(sub))
        segments.append(np.mean(sub))
    idx = int(EPISODES / 10)
    print("'Trained' (avg of last tenth) reward:", np.mean(totalRewards[-idx:]))
    print("Average reward:", np.mean(totalRewards))
    unique, counts = np.unique(np.array(totalRewards), return_counts=True) 
    print(dict(zip(unique, counts)))
    print("Total runtime: ", round(time.time() - start_time, 2), " seconds")
    np.save("dq_last_rewards", totalRewards)
    plt.plot(segments)
    plt.show()
