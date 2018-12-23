import gym
import gym_settemezzo
import matplotlib.pyplot as plt

env = gym.make('Settemezzo-v0')
env.reset()

num_rounds = 1000 # Payout calculated over num_rounds
num_samples = 1000 # num_rounds simulated over num_samples

average_payouts = []

for sample in range(num_samples):
    round = 1
    total_payout = 0 # to store total payout over 'num_rounds'
    
    while round <= num_rounds:
        action = env.action_space.sample()  # take random action 
        obs, payout, is_done, _ = env.step(action)
        total_payout += payout
        if is_done:
            env.reset() # Environment deals new cards to player and dealer
            round += 1
    average_payouts.append(total_payout)

plt.plot(average_payouts)                
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show()    
print ("Average payout after {}  rounds is {}".format(num_rounds, sum(average_payouts)/num_samples))

def normal_strategy(player_sum, dealer_up_card):
    """
    Returns appropriate action from a 2D array storing actions
    Actions obtained from figure 11 here- https://pdfs.semanticscholar.org/e1dd/06616e2d18179da7a3643cb3faab95222c8b.pdf
    Each row corresponds to player sum- from 2 to 21
    Each column corresponds to dealer_up_card- from 1 to 10
    Actions 1=hit 0=stay
    """
    actions = [[1]*8]*3 # 0.5 to 1.5
    actions.append([1]*2 + [0]*3 + [1]*3) #2
    actions.append([1]*3 + [0]*2 + [1]*3) #2.5
    actions.append([1] + [0]*4 + [1]*3) #3
    actions.append([1]*2 + [0]*3 + [1]*3) #3.5
    actions.append([0]*5 + [1]*3) #4
    actions.append([1]*2 + [0]*3 + [1]*3) #4.5
    actions.append([0]*5 + [1]*3) #5
    actions.append([0]*6 + [1]*2) #5.5
    actions.append([0]*7 + [1]*1) # 6
    actions.append([0]*7 + [1]*1) #6.5
    actions.extend([[0]*8]*2) # 7 to 7.5
     
    # dealer_up_card-2 takes care of input 1 which correcly looks up last column
    return actions[player_sum-1][dealer_up_card-1]

# Make sure actions have been stored correctly mainly when dealer's upcard is A

num_rounds = 1000 # Payout calculated over num_rounds
num_samples = 100 # num_rounds simulated over num_samples
total_payout = 0 # to store total payout over 'num_rounds'

for _ in range(num_samples):
    round = 1
    while round <= num_rounds:
        player_sum, dealer_up_card, is_done = (env._get_obs())
        
        # Take action based on normal strategy stored above
        action = normal_strategy(player_sum, dealer_up_card)
        
        obs, payout, is_done, _ = env.step(action)
        total_payout += payout
        if is_done:
            env.reset() # Environment deals new cards to player and dealer
            round += 1
    
print ("Average payout after {} rounds is {}".format(num_rounds, total_payout/num_samples))

