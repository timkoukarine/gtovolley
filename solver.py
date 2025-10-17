"""
By: Tim Koukarine

this module allows the user to solve for nash equilibrium in both perfect and
inperfect information games involving 2 players.

Order of Running:
Initialize a game with:

game = Game(num_subgames: int, A1: tuple[str], A2: tuple[str], NumT1: int, NumT2: int)

game.build_subgames() -> will ask for inputs for payoffs in each field for each player

game.set_game_weights() -> will ask for game probabilities (numerator, then denominator)

game.get_weighted_payoffs() - No Input Needed

game.set_weighted_types() -> Will ask for each player's realized type vector for each game

game.build_strategic_form() - No Input Needed

game.find_nash() -> will return pure nash equilibrium strategy profiles (if found)

check any game properties with:
game.subgames
game.strategic_game
game.{property... etc.}
"""
import pandas as pd
import numpy as np
from fractions import Fraction
from itertools import product

class Game:

    def __init__(self, num_subgames: int, A1: tuple[str], A2: tuple[str], T1: int, T2: int) -> None:

        self.num_subgames = num_subgames
        self.A1 = A1
        self.A2 = A2
        self.T1 = T1
        self.T2 = T2

        if self.T1 > 1:
            self.strategies1 = list(product(self.A1, repeat=len(range(self.T1))))
        else: 
            self.strategies1 = list(A1)

        if self.T2 > 1: 
            self.strategies2 = list(product(self.A2, repeat=len(range(self.T2))))
        else:
            self.strategies2 = list(A2)
        
        self.subgames = []
        self.game_weights = []
        self.weighted_payoffs = []
        self.weighted_game_types = {}
        self.strategic_game = []

    def build_subgames(self, subgames: list[pd.DataFrame] | None = None) -> None:
        if subgames: 
            self.subgames = subgames
            return
        for game in range(self.num_subgames):
            subgame  = pd.DataFrame(index=self.A1, columns=self.A2)
            for action1 in self.A1:
                for action2 in self.A2:
                    p1 = int(input(f"enter the payoff for profile ({action1, action2}) for player 1 in game {game}:"))
                    p2 = int(input(f"enter the payoff for profile ({action1, action2}) for player 2 in game {game}:"))
                    subgame.loc[action1, action2] = (p1, p2)
            self.subgames.append(subgame)

    def set_game_weights(self, game_weights: list[Fraction] | None = None) -> None:
        if game_weights:
            self.game_weights = game_weights
            return
        for game in range(self.num_subgames):
            num = int(input(f'enter numerator for game {game}'))
            den = int(input(f'enter denominator for game {game}'))
            self.game_weights.append(Fraction(num, den))

    def get_weighted_payoffs(self):
        for count, game in enumerate(self.subgames):
            weighted_subgame = pd.DataFrame(index=self.A1, columns=self.A2)
            for action1 in game.index:
                for action2 in game.columns:
                    p1 = game.loc[action1, action2][0] * self.game_weights[count]
                    p2 = game.loc[action1, action2][1] * self.game_weights[count]
                    weighted_subgame.loc[action1, action2] = (p1, p2)
            self.weighted_payoffs.append(weighted_subgame)        

    def set_weighted_types(self, weighted_types: dict[tuple[int, int], pd.DataFrame] | None = None) -> None:
        if weighted_types:
            self.weighted_game_types = weighted_types
            return
        for count, game in enumerate(self.weighted_payoffs):
            t1 = int(input(f'enter p1 type for game {count}'))
            t2 = int(input(f'enter p2 type for game {count}'))
            if (t1, t2) not in self.weighted_game_types:
                self.weighted_game_types.update({(t1, t2): game})
            else: 
                weighted_sum = pd.DataFrame(index=self.A1, columns=self.A2)
                game_stored = self.weighted_game_types[(t1, t2)]
                new_game = game
                for col_indx, col in enumerate(game_stored):
                    for row_indx, payoff in enumerate(game_stored[col]):
                        p1 = game_stored.iloc[row_indx, col_indx][0] + new_game.iloc[row_indx, col_indx][0]
                        p2 = game_stored.iloc[row_indx, col_indx][1] + new_game.iloc[row_indx, col_indx][1]
                        weighted_sum.iloc[row_indx, col_indx] = (p1, p2)
                self.weighted_game_types.update({(t1, t2): weighted_sum})

    def build_strategic_form(self):
        """
        Constructs the strategic form of the game by aggregating the weighted payoffs
        for each strategy profile of both players.
        """
        strategic_game = pd.DataFrame(index=self.strategies1, columns=self.strategies2)
        for strat1 in self.strategies1:
            for strat2 in self.strategies2:
                p1 = 0
                p2 = 0
                for key in self.weighted_game_types.keys():
                    prof1 = key[0]
                    prof2 = key[1]

                    p1 += self.weighted_game_types[key].loc[strat1[prof1 - 1], strat2[prof2 - 1]][0]
                    p2 += self.weighted_game_types[key].loc[strat1[prof1 - 1], strat2[prof2 - 1]][1]

                strategic_game.at[strat1, strat2] = (p1, p2)    
        self.strategic_game = strategic_game

    def find_nash(self) -> list[object]:
        nash_equilibriums = []
        max_pay1 = self.get_p1_max_payoffs()
        max_pay2 = self.get_p2_max_payoffs()
        for i, x in enumerate(max_pay2):
            for indx1 in x:
                for indx2 in max_pay1[indx1]:
                    if indx2 == i:
                        strat1 = self.strategies1[indx2]
                        strat2 = self.strategies2[indx1]
                        nash_equilibriums.append((strat1, strat2))
        return nash_equilibriums

    def get_p1_max_payoffs(self):
        max_pay1 = []
        for a in self.strategic_game.to_numpy().T:
            mx = float('-inf')
            inner = []
            for i, b in enumerate(a):
                if b[0] > mx:
                    inner = [i]
                    mx = b[0]
                elif b[0] == mx:
                    inner.append(i)
            max_pay1.append(inner)
        return max_pay1
    
    def get_p2_max_payoffs(self):
        max_pay2 = []
        for a in self.strategic_game.to_numpy():
            mx = float('-inf')
            inner = []
            for i, b in enumerate(a):
                if b[1] > mx:
                    inner = [i]
                    mx = b[1]
                elif b[1] == mx:
                    inner.append(i)
            max_pay2.append(inner)
        return max_pay2                    


# Quick Init for testing:
if __name__ == "__main__":
    game = Game(4, ("T", "B"), ("L", "R"), 2, 2)

    subgames = [] 
    for g in range(game.num_subgames):
            subgame  = pd.DataFrame(index=game.A1, columns=game.A2)
            for action1 in game.A1:
                for action2 in game.A2:
                    payoff = (np.random.randint(-10, 10), np.random.randint(1, 10))
                    subgame.loc[action1, action2] = payoff
            subgames.append(subgame)
    game.build_subgames(subgames=subgames)

    game.set_game_weights([Fraction(1, 3), Fraction(1, 6), Fraction(1, 4), Fraction(1, 4)])
    
    game.get_weighted_payoffs()
    
    weighted_types = {(1, 1): subgames[0], (1, 2): subgames[1], (2, 1): subgames[2], (2, 2): subgames[3]}
    game.set_weighted_types(weighted_types=weighted_types)

    game.build_strategic_form()