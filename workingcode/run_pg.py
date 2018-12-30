import gym
import gym_settemezzo
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time

env = gym.make('Settemezzo-v0')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
#env.seed(1985)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)

EPISODES = 1000000  # min 100, multiple of 100
rewards = []
totalReports = 10  # how often to report episode# and other details (ex. save ckpt). Default 10 times per run
reportingFactor = EPISODES / totalReports

if __name__ == "__main__":


    start_time = time.time()  # for tracking run time of method

    # Load checkpoint
    load_path = None # "pg_agent/last_agent.ckpt"  # None # "pg_agent/million_trained_agent.ckpt"
    save_path = "pg_agent/last_agent.ckpt"  # None #"output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = 2,  # observation space
        n_y = 2,  # action space
        learning_rate=0.0001,
        reward_decay=0.3,
        load_path=load_path,
        save_path=None
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

                # 4. Train neural network (OPTIONAL: and save agent)
                if episode % reportingFactor == 0:
                    PG.save_path = save_path
                
                discounted_episode_rewards = PG.learn()

                if episode % reportingFactor == 0:
                    PG.save_path = None
                    print("Observation:", observation, " Outcome:", observation_, " Reward:", reward)
                    print("Discounted episode reward stack (last episode): ", discounted_episode_rewards)

                break

            # Save new observation
            observation = observation_

    # Summary printouts of performance
    segments = []
    split = np.split(np.array(rewards), 100)
    print("Progression of segmented average scores: ")
    for sub in split:
        # print(sub)
        print(np.mean(sub))
        segments.append(np.mean(sub))

    idx = int(EPISODES / 10)
    print("'Trained' (average of last tenth) reward:", np.mean(rewards[-idx:]))
    print("Overall average reward:", np.mean(rewards))
    unique, counts = np.unique(np.array(rewards), return_counts=True) 
    print(dict(zip(unique, counts)))
    print("Total runtime: ", round(time.time() - start_time, 2), " seconds")
    np.save("pg_last_rewards", rewards)
    plt.plot(segments)
    plt.show()
