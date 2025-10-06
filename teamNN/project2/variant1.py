# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearn_character import ApproxQLearningCharacter

agent = ApproxQLearningCharacter(name="qhero", avatar="Q")

# Create the game
g = Game.fromfile('map.txt')

# TODO Add your character
g.add_character(agent)

# Run!
g.go()
