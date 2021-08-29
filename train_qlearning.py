from support_functions import *
import numpy as np
import dill
import itertools
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import copy
from pathlib import Path
import subprocess
from datetime import datetime

start = datetime.now()

# Force the INFO messages to be printed to the console
logging.basicConfig(level=logging.DEBUG)

# Loading our tree opponent from the first article
logging.info('Checking if pkl files are generated...')
if not Path('tree_tactoe_3x3.pkl').exists():
    subprocess.call('generate_tree.py')
else:
    logging.info('tree_tactoe_3x3.pkl found...')

logging.info('Loading tree...')
with open('tree_tactoe_3x3.pkl', 'rb') as f:
    tree = dill.load(f)

# Using memoization to speed up the tree
logging.info('Precomputing tree moves...')
precompute_tree_moves(tree)

# Create the board and player
tictactoe = Tictoe(3)
player_tree = Player(1, tree, alpha = 0.01,
                              gamma = 0.8,
                              epsilon = 0.1) 

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
logging.info('done (%s)...' % (datetime.now() - start))