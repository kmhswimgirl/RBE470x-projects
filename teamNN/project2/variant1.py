# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game
import random

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearn_character import ApproxQLearningCharacter, TRAINING

for i in range(100):  # number of training games
    random.seed() # TODO Change this if you want different random choices
    print(f"\n[Training Run {i+1}/50]")
    g = Game.fromfile('map.txt')
    agent = ApproxQLearningCharacter("me", "Q", 0, 0)
    g.add_character(agent)
    g.go(1)  # Run full game
    # Save weights after the game finishes
    if TRAINING:
        agent.save_weights()

