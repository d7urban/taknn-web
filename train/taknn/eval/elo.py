"""Elo match runner using Rust engine via PyO3."""

import math
import random
import tak_python


def calculate_elo(score_a, score_b, n_games):
    """Compute Elo difference and 95% confidence interval."""
    if n_games == 0:
        return 0, 0

    wa = score_a / n_games
    if wa == 0:
        return -float('inf'), 0
    if wa == 1:
        return float('inf'), 0

    elo_diff = -400 * math.log10(1 / wa - 1)

    std_err = math.sqrt(wa * (1 - wa) / n_games)
    ci = 1.96 * std_err * 400 / (math.log(10) * wa * (1 - wa))

    return elo_diff, ci


class RandomPlayer:
    """Plays a random legal move."""
    def pick_move(self, game):
        n = game.legal_move_count()
        return random.randint(0, n - 1)


class MatchRunner:
    def __init__(self, player_a, player_b, size=6, games=50):
        self.player_a = player_a
        self.player_b = player_b
        self.size = size
        self.games = games

    def run(self):
        results = []
        for i in range(self.games):
            if i % 2 == 0:
                res = self._play_game(self.player_a, self.player_b)
                results.append(res)
            else:
                res = self._play_game(self.player_b, self.player_a)
                results.append(1 - res)

        score_a = sum(results)
        score_b = self.games - score_a
        return calculate_elo(score_a, score_b, self.games)

    def _play_game(self, white, black):
        """Play one game. Returns 1 if white wins, 0.5 draw, 0 if black wins."""
        game = tak_python.PyGameState(self.size)
        players = [white, black]

        while not game.is_terminal() and game.ply() < 200:
            player = players[game.side_to_move()]
            move_idx = player.pick_move(game)
            game.apply_move(move_idx)

        result = game.result()
        if result in (0, 2):  # White road/flat win
            return 1
        elif result in (1, 3):  # Black road/flat win
            return 0
        else:
            return 0.5
