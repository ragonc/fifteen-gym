import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return float(a > b) - float(a < b)


# 0.5 = Ace, 2-7 = Number cards, Jack/Queen/King = 0.5
deck = [2, 4, 6, 8, 10, 12, 14, 1, 1, 1]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    # print('drawing hand')
    return [draw_card(np_random)]


# def usable_ace(hand):  # Does this hand have a usable ace?
# return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    # if usable_ace(hand):
    # return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 15


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 14]


class SettemezzoEnv(gym.Env):
    """
        Simple 7mezzo environment
        Blackjack is a card game where the goal is to obtain cards that sum to as
        near as possible to 7.5 without going over.  They're playing against a fixed
        dealer.

        Face cards (Jack, Queen, King) have point value 0.5.
        Aces counted as 1.
        This game is placed with an infinite deck (or with replacement).
        The game starts with each (player and dealer) having one face up and one
        face down card.

        The player can request additional cards (hit=1) until they decide to stop
        (stick=0) or exceed 7.5 (bust).

        After the player sticks, the dealer reveals their facedown card, and draws
        until their sum is 17 or greater.  If the dealer goes bust the player wins.
        If neither player nor dealer busts, the outcome (win, lose, draw) is
        decided by whose sum is closer to 21.  The reward for winning is +1,
        drawing is 0, and losing is -1.

        The observation of a 2-tuple of: the players current sum,
        the dealer's one showing card (0.5-7).

       This environment corresponds to the version of the blackjack problem
        described in Example 5.1 in Reinforcement Learning: An Introduction
        by Sutton and Barto.
        http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(29),  # 0.5 to 7 + 7
            spaces.Discrete(15)))  # 0.5 to 7
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack (settemezzo) win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 10:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0])

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()
