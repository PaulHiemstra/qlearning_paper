from support_functions import *
import numpy as np
import dill
import itertools
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import copy

# Force the INFO messages to be printed to the console
logging.basicConfig(level=logging.DEBUG)

logging.info('Loading tree...')
with open('tree_tactoe_3x3.pkl', 'rb') as f:
    tree = dill.load(f)

logging.info('Precomputing best moves...')
all_states = []
for length in range(1,9):
    tree_states = [''.join(state) for state in list(itertools.permutations(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], r=length))]
    all_states.extend(tree_states)

for state in tqdm(all_states):
    try:
        move = determine_move(tree, state, False) 
    except:
        pass 

tictactoe = Tictoe(3)
epsilon_value = 0.1
player_tree = Player(1,
                      tree, 
                      alpha = 0.01,
                      gamma = 0.8,
                      epsilon = epsilon_value)

logging.info('Starting the training loop...')
no_episodes = int(sys.argv[1])     # Get the number of episodes to run from the input args
for ep_idx in tqdm(range(no_episodes)):
    while not tictactoe.is_endstate():
        tictactoe = player_tree.make_move(tictactoe)
        tictactoe = player_tree.make_computer_move(tictactoe)
        player_tree.update_qtable()
    tictactoe.reset_board()

logging.info('Saving the model...')
with open('trained_player.pkl', 'wb') as f:
    dill.dump(player_tree, f)

logging.info('Training finished...')
logging.info('done...')