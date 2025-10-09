# This is necessary to find the main code
import sys
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')
sys.path.insert(1, '../teamNN')

# Import necessary stuff
from game import Game
import random

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from FakeCharacter import ApproxQLearningCharacter, TRAINING
from monsters.stupid_monster import StupidMonster

for i in range(10):  # number of training games
    random.seed() # TODO Change this if you want different random choices
    print(f"\n[Training Run {i+1}/50]")
    g = Game.fromfile('/Users/dhruvmadan/RBE4701/RBE470x-projects/teamNN/map.txt')
    agent = ApproxQLearningCharacter("me", "Q", 0, 0)
    g.add_character(agent)
    g.go(1)  # Run full game
    # Save weights after the game finishes
    if TRAINING:
        agent.save_weights()

