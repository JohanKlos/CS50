"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def main():
    board = initial_state()


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # we have the board, so we determine the number actions allowed
    # if that number is EVEN, then it's O's turn
    # if that number is ODD, then it's X's turn
    allowed_actions = actions(board)

    # https://book.pythontips.com/en/latest/ternary_operators.html
    return O if len(allowed_actions)%2==0 else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    allowed_actions = []
    # loop over the array and see which positions are EMPTY
    for i, row in enumerate(board):
        for j, col in enumerate(row):
            if col is EMPTY:
                allowed_actions.append((i, j))

    return allowed_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # make a deep-copy of the board, the AI program might want to go back
    # een slice new_board = board[:][:]
    new_board = copy.deepcopy(board)
    # new_board = [ele[:] for ele in board]
    # determines allowed actions for new_board
    new_allowed_actions = actions(new_board)
    if action in new_allowed_actions:
        # change new_board
        new_board[action[0]][action[1]] = player(new_board)
        return new_board
    else:
        raise Exception("That action is not possible.")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check if there's three X's on the board - if not, no winner
    if (sum(x.count("X") for x in board)) < 3:
        return None

    # diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    # rows
    elif board[0][0] == board[0][1] == board[0][2]:
        return board[0][0]
    elif board[1][0] == board[1][1] == board[1][2]:
        return board[1][0]
    elif board[2][0] == board[2][1] == board[2][2]:
        return board[2][0]
    # cols
    elif board[0][0] == board[1][0] == board[2][0]:
        return board[0][0]
    elif board[0][1] == board[1][1] == board[2][1]:
        return board[0][1]
    elif board[0][2] == board[1][2] == board[2][2]:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or len(actions(board)) == 0:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/
    https://en.wikipedia.org/wiki/Minimax
    """
    current_player = player(board)

    if current_player == X:
        # https://www.w3schools.com/python/ref_math_inf.asp
        # X is maximizing player
        v = -math.inf
        # loop over actions available on the current board
        for action in actions(board):
            if len(actions(board)) == 9:
                # first move, so just put the x in the center
                best_move = (1,1)
                break
            # get the min value of the result of an action on the current board
            k = min_v(result(board, action))
            # v is initially -math.inf
            if k > v:
                # whenever we find a better v, this is the new best_move
                v = k
                best_move = action
    else:
        # O is minimizing player
        v = math.inf
        for action in actions(board):
            k = max_v(result(board, action))
            if k < v:
                v = k
                best_move = action
    return best_move


def max_v(board):
    """
    get the max value of the result of an action on the current board
    """
    # if the board is "terminal" we stop this wargame
    if terminal(board):
        # and we return who has won
        return utility(board)
    # but the game is on, so we set the worst case v
    v = -math.inf
    for action in actions(board):
        # and we determine per possible action what the new max move is
        v = max(v, min_v(result(board, action)))
    return v


def min_v(board):
    """
    get the min value of the result of an action on the current board
    """
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_v(result(board, action)))
    return v


if __name__ == "__main__":
    main()
