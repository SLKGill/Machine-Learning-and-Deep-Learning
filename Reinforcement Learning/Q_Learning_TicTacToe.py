import random
# do not need any machine learning algorithms or neural networks because represent Q function using data structure (dictionary)

BLANK = ' '  # empty cell
AI_PLAYER = 'X'  # computer is X
HUMAN_PLAYER = 'O'  # human is O
TRAINING_EPOCHS = 5
# during training procedure, need to make several training iterations/epochs
# need 2 AI players, AI PLAYER 1 and AI PLAYER 2, who will compete with each other, and play 40000 gameplays to learn the Q function with reinforcement learning
TRAINING_EPSILON = 0.4  # related to exploration vs. exploitation, AI will try new moves with 40% probaility
REWARD_WIN = 10
REWARD_LOSE = -10
REWARD_TIE = 0  # or less than 0 like -3 so AI player aims to win, wants to avoid tie


class Player:  # for both human and AI player

    @staticmethod  # used since this is a static method
    def show_board(board):
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))


# inheritance from parent class Player, HumanPlayer is a Player
class HumanPlayer(Player):

    def reward(self, value, board):
        pass  # not dealing with rewarwds with human

    def make_move(self, board):  # board size is 9
        while True:
            try:
                self.show_board(board)  # show board to user
                # allow user to make input, waiting for input from user
                move = input("Your next move (cell index 1-9):")  # input must be one of the options
                move = int(move)

                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move-1  # to be compatible with the array board


# inheritance from parent class Player, AIPlayer is a Player
class AIPlayer(Player):

    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9, default_q=1):
        # gamma being 0.9 means rewards in the future have smaller value than rewards at the actual step
        # if alpha=1 end up with bellman equation
        self.EPSILON = epsilon  # this is the epsilon parameter of the model: the probability of exploration
        self.ALPHA = alpha  # learning rate
        # discount parameter for future reward (rewards now are better than rewards in the future)
        self.GAMMA = gamma
        # if the given move at the given state is not defined yet, we have a default Q value (1)
        self.DEFAULT_Q = default_q
        # in the beginning the Q dictionary is empty which means the Q function will not be working well
        # during training calculating Q values and storing them in the dictionary, may happen given move at state is not defined so apply default_q

        # Q(s,a) function is a dictionary in this implementation. This is the Q function - Q SxA -> R

        self.q = {}  # return a value for s state and a action (s,a) pair (key value pairs)
        self.move = None  # previous move during the game
        # board in the previous iteration, actual state of the board, 9 empty strings in the beginning
        self.board = (' ',) * 9

    # empty cells on the grid (board)
    def available_moves(self, board):
        return [i for i in range(9) if board[i] == ' ']  # size of available cells eill decrease

    # Q(s,a) -> Q value for (s,a) pairs if no Q value exists then create a new one with the default value (1), otherwise return q value present in dictionary
    def get_q(self, state, action):
        if self.q.get((state, action)) is None:  # not in q table means create a new one
            self.q[(state, action)] = self.DEFAULT_Q

        return self.q[(state, action)]  # value associated with state and action

    # make a random move with epsilon probability (exploration) or pick the action with the highest Q value (exploitation)
    # tune with EPSILON parameter (40% probabilty exploring a new value and 60% probabilty pick the action with higheset Q value (exploitation))
    def make_move(self, board):
        self.board = tuple(board)  # list is mutable can be changed, tuple is immutable
        actions = self.available_moves(board)  # 1d array with empty cell indexes

        # action with epsilon probability
        if random.random() < self.EPSILON:
            # this is in index (0-8 board cell related index)
            self.move = random.choice(actions)
            return self.move

        # take the action with the highest Q values
        q_values = [self.get_q(self.board, a) for a in actions]
        max_q_value = max(q_values)  # may be multiple max q values

        # if multiple best actions, choose one at random
        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
            best_move = actions[random.choice(best_actions)]
        # there is just a single best move (best action)
        else:
            best_move = actions[q_values.index(max_q_value)]

        self.move = best_move  # best move is that of highest q value
        return self.move

        # reward functions
        # lets evaluate a given state, so update the Q(s,a) table regarding s state and a action
        # need maximum of next possible states
    def reward(self, reward, board):
        if self.move:
            prev_q = self.get_q(self.board, self.move)
            max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.board)])
            self.q[(self.board, self.move)] = prev_q + self.ALPHA * \
                (reward + self.GAMMA * max_q_new - prev_q)
            # the equation for Q learning                                                             self.GAMMA * max_q_new - prev_q)


class TicTacToe:
    def __init__(self, player1, player2):  # 2 AI players competing each other during training.
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 9

    def play(self):

        # game loop
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                # in training not actually a human player just for naming
                player_tickers = (AI_PLAYER, HUMAN_PLAYER)
            else:
                player = self.player2
                other_player = self.player1
                player_tickers = (HUMAN_PLAYER, AI_PLAYER)

            # check state of the game (win, loose, draw)
            game_over, winner = self.is_game_over(player_tickers)

            # game is over, distribute rewarwds
            if game_over:
                if winner == player_tickers[0]:
                    player.show_board(self.board[:])  # all values
                    print('%s won!' % player.__class__.__name__, '\n')
                    player.reward(REWARD_WIN, self.board[:])
                    other_player.reward(REWARD_LOSE, self.board[:])
                if winner == player_tickers[1]:
                    player.show_board(self.board[:])  # all values
                    print('%s won!' % other_player.__class__.__name__, '\n')
                    player.reward(REWARD_LOSE, self.board[:])
                    other_player.reward(REWARD_WIN, self.board[:])
                else:
                    player.show_board(self.board[:])  # all values
                    print('Tie!\n')
                    player.reward(REWARD_TIE, self.board[:])
                    other_player.reward(REWARD_TIE, self.board[:])
                break

            # next player's turn in the next iteration
            self.first_player_turn = not self.first_player_turn

            # actual player's best move (based on Q(s,a) table for AI player)
            move = player.make_move(self.board)
            self.board[move] = player_tickers[0]

    def is_game_over(self, player_tickers):
        # consider both players (X and O players - these are the tickers)
        for player_ticker in player_tickers:
            # check horizontal dimension (rows)
            for i in range(3):
                if self.board[3 * i + 0] == player_ticker and self.board[3 * i + 1] == player_ticker and self.board[3 * i + 2] == player_ticker:
                    return True, player_ticker

            # check vertical dimension (columns)
            for j in range(3):
                if self.board[j + 0] == player_ticker and self.board[j + 3] == player_ticker and self.board[j + 6] == player_ticker:
                    return True, player_ticker

            # check diagonal dimensions (top left to bottom right and top right to bottom left)
            if self.board[0] == player_ticker and self.board[4] == player_ticker and self.board[8] == player_ticker:
                return True, player_ticker

            if self.board[2] == player_ticker and self.board[4] == player_ticker and self.board[6] == player_ticker:
                return True, player_ticker

        # draw cases
        if self.board.count(' ') == 0:
            return True, None
        else:
            return False, None  # game not over


if __name__ == '__main__':
    ai_player_1 = AIPlayer()
    ai_player_2 = AIPlayer()

    print('Training the AI player(s)...\n\n')

    ai_player_1.EPSILON = TRAINING_EPSILON
    ai_player_2.EPSILON = TRAINING_EPSILON

    for _ in range(TRAINING_EPOCHS):
        game = TicTacToe(ai_player_1, ai_player_2)
        game.play()

    print('\nTraining is done\n\n')

    # epsilon=0 means no exploration it will use the Q(s,a) function to make the moves
    # it will use the Q function to make moves, so reinitialize epsilon to be 0
    ai_player_1.EPSILON = 0
    human_player = HumanPlayer()
    game = TicTacToe(ai_player_1, human_player)
    game.play()
