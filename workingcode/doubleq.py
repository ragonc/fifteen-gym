
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_settemezzo


def maxAction(Q1, Q2, state):
    #values = np.array([Q1[state, a] + Q2[state, a] for a in range(2)])
    #print(values)
    hit = Q1[state, 1] + Q2[state, 1]   
    stick = Q1[state, 0] + Q2[state, 0]
    print('state number: ' + str(state))
    print('hit: ' + str(hit))
    print('stick: ' + str(stick))
    # action = np.argmax(values)
    return int(hit > stick)


# takes the raw player sum value and dealer show card value
# and returns the indices of the matching state
def getState(observation):
    #playerSumSpace = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
    #                  6, 6.5, 7]  # for us this will be 0.5 - 14 or 1 - 28 ?
    #dealerShowSpace = [0.5, 1, 2, 3, 4, 5, 6, 7]  # for us this will be 0 (1) to 14
    # playerSumSpace = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #                   12, 13, 14]  # for us this will be 0.5 - 14 or 1 - 28 ?
    playerSumSpace = list(range(1, 30))  # up to 7 + 7 (28) however, will still need to learn to stay on 15/7.5 :)
    dealerShowSpace = [1, 2, 4, 6, 8, 10, 12, 14]  # for us this will be 0 (1) to 14
    player, dealer = observation
    state = (playerSumSpace.index(player) * 8) + dealerShowSpace.index(dealer)
    print('obs: ' + str(player) + ' ' + str(dealer))
    print('calculated state: ' + str(state))
    # player = playerSumSpace.index(player)
    # dealer = dealerShowSpace.index(dealer)

    return state  # NOTE: should return index of player * dealer.size + dealer


def plotRunningAverage(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    env = gym.make('Settemezzo-v0')
    # model hyperparameters
    ALPHA = 0.1  # learning rate
    GAMMA = 0.9  # discount factor
    EPS = 1.0  # exploration rate
    epsRF = 0.999  # per-game exploration reduction factor

    # player possibilities: 14; dealer poss: 8
    # states = np.zeros((14, 8))
    # Q1 = np.zeros((29, 8, 2))
    # Q2 = np.zeros((29, 8, 2))
    Q1 = np.zeros((30 * 8, 2))
    Q2 = np.zeros((30 * 8, 2))

    # Q1, Q2 = {}, {}
    # for s in range(states.size):
    #     for a in range(2):
    #         Q1[s, a] = 0
    #         Q2[s, a] = 0

    numGames = 30000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
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
        # EPS -= 2 / (numGames) if EPS > 0 else 0
        if EPS > 0.01:
            EPS = EPS * epsRF
        # print(EPS)
        # print(epRewards)
        totalRewards[i] = epRewards

    # plt.plot(totalRewards, 'b--')
    plt.show()
    # print(states)
    # print('alpha is: ' + str(ALPHA))
    # print('gamma is: ' + str(GAMMA))
    # print('eps is: ' + str(EPS))
    # print(Q1)
    # print(Q2)
    print(np.mean(totalRewards))
    plotRunningAverage(totalRewards)