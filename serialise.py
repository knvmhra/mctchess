import numpy as np
import chess
import chess.pgn
from constants import POLICY_MAP

_EMPTY_STATE = np.zeros(shape= (42, 8, 8))

def u64_to_arr(bb: int):
    if bb == 0:
        return np.zeros((8, 8))
    elif bb == 0xffffffffffffffff:
        return np.ones((8, 8))
    else: 
        return np.array([int((bb & (1 << i)) != 0) for i in range(64)]).reshape((8, 8))
    
def _serialise_state(board: chess.Board):
    state: list[int] = []

    turn = chess.BB_ALL if board.turn == chess.WHITE else chess.BB_EMPTY
    state.append(turn)

    #awkward code but order matters
    for p in chess.PIECE_TYPES:
        wbb = board.pieces_mask(p, True)
        state.append(wbb)

    for p in chess.PIECE_TYPES:
        bbb = board.pieces_mask(p, False)
        state.append(bbb)

    if board.has_kingside_castling_rights(chess.WHITE):
        castle_bb |= chess.BB_G1
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_bb |= chess.BB_C1 
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_bb |= chess.BB_G8 
    if board.has_queenside_castling_rights(chess.BLACK):
        castle_bb |= chess.BB_C8
    state.append(castle_bb)

    state = [u64_to_arr(s) for s in state]
    return state

def serialise_board(board: chess.Board, len_history: 3):
    states = []
    if len_history == 1:
        states.append(_serialise_state)
        return np.stack(states, 0)
    
    board = board.copy()
    for _ in range(len_history): #also encode previous board states
        if len(board.move_stack) < len_history:
            states.extend(_EMPTY_STATE) #empty if not enough history
            continue
        states.extend(_serialise_state(board))
        board.pop()
    return np.stack(states, 0) #across the zeroth axis so board state is C, H, W. torch uses N, C, H, W


def serialise_move(mv: chess.Move):
    mv: str = mv.uci()
    hot = POLICY_MAP.index(mv)
    out = np.zeros(1968) #len(policy_arr) is 1968
    out[hot] = 1
    return out

def generate_data(pgn_file: str, num_games: int =3):
    states, moves, results = [], [], []
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    pgn = open(pgn_file)
    
    for i in range(num_games):
        game = chess.pgn.read_game(pgn)
        if not game:
            break
        res = game.headers['Result']
        termination = game.headers['Termination']
        if res not in values or termination != 'Normal':
            continue
        v = values[res]
        board = game.board()
        for move in game.mainline_moves():
            s = serialise_board(board)
            states.append(s)
            m = serialise_move(move)
            moves.append(m)
            results.append(v) 
            board.push(move)
        if i % 5000 == 0:
            print(f"Parsed {i} games")
    X = np.array(states)
    P = np.array(moves)
    V = np.array(results)
    return X, P, V

if __name__ == "__main__":
    X, P, V = generate_data("/Users/kmwork/Desktop/lichess_elite_2020-06.pgn", 25000)
    np.savez("datasets/v1mini_25000.npz", X, P, V)
