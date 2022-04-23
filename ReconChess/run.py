import util
import config_loader
from play_game import *
config_loader.load_config()
from util import ENV

wins = 0

for i in range(ENV.num_games):
    name_one, constructor_one = load_player(ENV.white)
    if len(inspect.signature(constructor_one.__init__).parameters) == len(ENV.white_args) + 1:
        player_one = constructor_one(*ENV.white_args)
    else:
        player_one = constructor_one()

    name_two, constructor_two = load_player(ENV.black)
    if len(inspect.signature(constructor_two.__init__).parameters) == len(ENV.black_args) + 1:
        player_two = constructor_two(*ENV.black_args)
    else:
        player_two = constructor_two()

    players = [player_one, player_two]
    player_names = [name_one, name_two]

    if name_one == "Human":
        color = input("Play as (0)Random (1)White (2)Black: ")
        if color == '2' or (color == '0' and random.uniform(0, 1) < 0.5):
            players.reverse()
            player_names.reverse()

    win_color, win_reason = play_local_game(players[0], players[1], player_names, verbose=False)
    if win_color == chess.WHITE:
        wins += 1
    print('Game Over!')
    if win_color is not None:
        print(win_reason)
    else:
        print('Draw!')
print(f"WHITE win rate: {wins / ENV.num_games}")
