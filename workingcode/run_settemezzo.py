import gym
import gym_settemezzo
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Settemezzo-v0')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
#env.seed(11)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)

EPISODES = 100000  # min 100, multiple of 100
rewards = []
totalReports = 10  # how often to report episode number and other details. Default 10 times per run
reportingFactor = EPISODES / totalReports

if __name__ == "__main__":


    # Load checkpoint
    load_path = None #"output/weights/CartPole-v0.ckpt"
    save_path = None #"output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = 2,  # observation space
        n_y = 2,  # action space
        learning_rate=0.0001,
        reward_decay=0.5,
        load_path=load_path,
        save_path=save_path
    )

    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        if episode % reportingFactor == 0:
            print("Episode: ", episode)
            # print(PG.episode_actions)

        while True:
            # 1. Choose an action based on observation
            # player card sum, dealer showing card: 
            obs = np.asarray(observation)
            action = PG.choose_action(obs)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 3. Store transition for training
            PG.store_transition(observation, action, reward)

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)

                # print("==========================================")
                # print("Episode: ", episode)
                # print("Reward: ", episode_rewards_sum)

                # 4. Train neural network
                discounted_episode_rewards = PG.learn()

                if episode % reportingFactor == 0:
                    print("der returned", discounted_episode_rewards)

                break

            # Save new observation
            observation = observation_

    # PG.plot_cost()
    idx = int(EPISODES / 100)
    print("Average 'Trained' reward:", np.mean(rewards[-idx:]))

    unique, counts = np.unique(np.array(rewards), return_counts=True) 
    print(dict(zip(unique, counts)))

    tenned = []
    split = np.split(np.array(rewards), 100)
    for sub in split:
        # print(sub)
        print(np.mean(sub))
        tenned.append(np.mean(sub))

    plt.plot(tenned)
    plt.show()
