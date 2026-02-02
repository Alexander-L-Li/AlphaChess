import chess
import numpy as np
import torch

HISTORY_LENGTH = 8

INPUT_PLANES = 119


class ActionMapper:
    """
    Maps chess moves to integer indices and vice versa for neural network I/O.
    Creates a vocabulary of all possible chess moves (source square to destination
    square), including underpromotions to rook, bishop, and knight. Queen promotions
    use the standard move encoding. The vocabulary covers approximately 4,000 moves.
    """
    def __init__(self):
        self.move_to_id = {}
        self.id_to_move = {}
        self._generate_vocabulary()
        self.vocab_size = len(self.move_to_id)

    def _generate_vocabulary(self):
        """
        Generate all possible moves from square to square.
        This covers ~4000 moves (including castling, en passant)
        """
        idx = 0
        for src in range(64):
            for dst in range(64):
                if src == dst: continue

                # Standard move
                move = chess.Move(src, dst)
                self._add_move(move, idx)
                idx += 1

                # Promotion move
                is_promotion_sq = (
                    (chess.square_rank(dst) == 7 and chess.square_rank(src) == 6) or
                    (chess.square_rank(dst) == 0 and chess.square_rank(src) == 1)
                )

                if is_promotion_sq:
                    for promotion_piece in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        prom_move = chess.Move(src, dst, promotion=promotion_piece)
                        self._add_move(prom_move, idx)
                        idx += 1

    def _add_move(self, move, idx):
        uci = move.uci()
        if uci not in self.move_to_id:
            self.move_to_id[uci] = idx
            self.id_to_move[idx] = move

    def encode(self, move):
        return self.move_to_id.get(move.uci())

    def decode(self, idx):
        return self.id_to_move.get(idx)


def _get_piece_planes(board):
    """
    Generate 12 planes for piece positions (6 piece types × 2 colors).

    Returns planes from the perspective of the current player:
    - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
    - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
    """
    pieces = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]
    layers = []

    # Determine colors from current player's perspective
    if board.turn == chess.WHITE:
        current_color, opponent_color = chess.WHITE, chess.BLACK
    else:
        current_color, opponent_color = chess.BLACK, chess.WHITE

    # Current player's pieces
    for piece_type in pieces:
        layer = np.zeros((8, 8), dtype=np.float32)
        for square in board.pieces(piece_type, current_color):
            rank, file = chess.square_rank(square), chess.square_file(square)
            layer[rank, file] = 1.0
        layers.append(layer)

    # Opponent's pieces
    for piece_type in pieces:
        layer = np.zeros((8, 8), dtype=np.float32)
        for square in board.pieces(piece_type, opponent_color):
            rank, file = chess.square_rank(square), chess.square_file(square)
            layer[rank, file] = 1.0
        layers.append(layer)

    return layers


def _get_repetition_planes(board):
    """
    Generate 2 planes indicating position repetition count.
    - Plane 0: All 1s if position has occurred once before
    - Plane 1: All 1s if position has occurred twice before (about to be 3-fold)
    """
    # Count how many times this position has occurred
    repetitions = 0

    # Check transposition table if available, otherwise count manually
    if board.is_repetition(2):
        repetitions = 2
    elif board.is_repetition(1):
        repetitions = 1

    plane1 = np.ones((8, 8), dtype=np.float32) if repetitions >= 1 else np.zeros((8, 8), dtype=np.float32)
    plane2 = np.ones((8, 8), dtype=np.float32) if repetitions >= 2 else np.zeros((8, 8), dtype=np.float32)

    return [plane1, plane2]


def _get_castling_planes(board):
    """
    Generate 4 planes for castling rights (from current player's perspective).
    - Plane 0: Current player kingside
    - Plane 1: Current player queenside
    - Plane 2: Opponent kingside
    - Plane 3: Opponent queenside
    """
    if board.turn == chess.WHITE:
        current_k = board.has_kingside_castling_rights(chess.WHITE)
        current_q = board.has_queenside_castling_rights(chess.WHITE)
        opp_k = board.has_kingside_castling_rights(chess.BLACK)
        opp_q = board.has_queenside_castling_rights(chess.BLACK)
    else:
        current_k = board.has_kingside_castling_rights(chess.BLACK)
        current_q = board.has_queenside_castling_rights(chess.BLACK)
        opp_k = board.has_kingside_castling_rights(chess.WHITE)
        opp_q = board.has_queenside_castling_rights(chess.WHITE)

    planes = []
    for has_right in [current_k, current_q, opp_k, opp_q]:
        if has_right:
            planes.append(np.ones((8, 8), dtype=np.float32))
        else:
            planes.append(np.zeros((8, 8), dtype=np.float32))

    return planes


def _get_en_passant_plane(board):
    """
    Generate 1 plane marking the en passant square (if available).
    """
    plane = np.zeros((8, 8), dtype=np.float32)
    if board.has_legal_en_passant():
        ep_square = board.ep_square
        if ep_square is not None:
            rank, file = chess.square_rank(ep_square), chess.square_file(ep_square)
            plane[rank, file] = 1.0
    return [plane]


def _get_fifty_move_plane(board):
    """
    Generate 1 plane with normalized fifty-move counter.
    Value is halfmove_clock / 100 (clamped to [0, 1]).
    """
    normalized = min(board.halfmove_clock / 100.0, 1.0)
    plane = np.full((8, 8), normalized, dtype=np.float32)
    return [plane]


def _get_color_plane(board):
    """
    Generate 1 plane indicating side to move.
    All 1s if white to move, all 0s if black.
    """
    if board.turn == chess.WHITE:
        return [np.ones((8, 8), dtype=np.float32)]
    else:
        return [np.zeros((8, 8), dtype=np.float32)]


def board_to_tensor(board):
    """
    Convert a chess.Board to AlphaZero-style tensor representation.

    Creates a 119-plane tensor encoding the board state:

    History planes (96 planes = 12 × 8 time steps):
    - For each of 8 time steps (T, T-1, ..., T-7):
      - 6 planes for current player's pieces (P, N, B, R, Q, K)
      - 6 planes for opponent's pieces (P, N, B, R, Q, K)

    Auxiliary planes (23 planes):
    - 4 planes: Castling rights (current K, current Q, opponent K, opponent Q)
    - 1 plane: En passant square
    - 1 plane: Fifty-move counter (normalized)
    - 1 plane: Color to move
    - 2 planes: Repetition count (1x, 2x)
    - 14 planes: Padding/reserved (zeros) to reach 119

    All piece planes are from the perspective of the current player.

    Args:
        board: The chess.Board position to convert.

    Returns:
        torch.Tensor of shape (119, 8, 8) with float32 dtype.
    """
    layers = []

    # Build history by replaying the game
    # We need to go back through the move stack to get previous positions
    history_boards = []
    temp_board = board.copy()
    history_boards.append(temp_board.copy())

    # Get up to HISTORY_LENGTH-1 previous positions
    moves_to_undo = min(len(temp_board.move_stack), HISTORY_LENGTH - 1)
    for _ in range(moves_to_undo):
        temp_board.pop()
        history_boards.append(temp_board.copy())

    # Reverse so oldest is last (we'll iterate from current to oldest)
    history_boards = history_boards[::-1]

    # Pad with empty boards if we don't have enough history
    while len(history_boards) < HISTORY_LENGTH:
        empty_board = chess.Board(fen=None)  # Empty board
        history_boards.insert(0, empty_board)

    # Take most recent HISTORY_LENGTH boards (most recent first for the tensor)
    history_boards = history_boards[-HISTORY_LENGTH:][::-1]

    # Generate piece planes for each historical position (96 planes)
    for hist_board in history_boards:
        if hist_board.fen() == chess.Board(fen=None).fen():
            # Empty board - add 12 zero planes
            layers.extend([np.zeros((8, 8), dtype=np.float32) for _ in range(12)])
        else:
            layers.extend(_get_piece_planes(hist_board))

    # Auxiliary planes for current position (23 planes)
    layers.extend(_get_castling_planes(board))      # 4 planes
    layers.extend(_get_en_passant_plane(board))     # 1 plane
    layers.extend(_get_fifty_move_plane(board))     # 1 plane
    layers.extend(_get_color_plane(board))          # 1 plane
    layers.extend(_get_repetition_planes(board))    # 2 planes

    # Padding planes to reach 119 (14 zero planes)
    num_padding = INPUT_PLANES - len(layers)
    for _ in range(num_padding):
        layers.append(np.zeros((8, 8), dtype=np.float32))

    return torch.tensor(np.stack(layers), dtype=torch.float32)


def board_to_tensor_simple(board):
    """
    Simple 13-plane tensor encoding (legacy version for compatibility).

    Creates a 13-plane tensor encoding:
    - Planes 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    - Planes 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    - Plane 12: Turn indicator (all 1s if white to move, all 0s if black)

    Args:
        board: The chess.Board position to convert.

    Returns:
        torch.Tensor of shape (13, 8, 8) with float32 dtype.
    """
    pieces = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]
    layers = []

    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in pieces:
            layer = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece_type, color):
                rank, file = chess.square_rank(square), chess.square_file(square)
                layer[rank, file] = 1.0
            layers.append(layer)

    turn_layer = np.zeros((8, 8), dtype=np.float32)
    if board.turn == chess.WHITE:
        turn_layer.fill(1.0)
    layers.append(turn_layer)

    return torch.tensor(np.stack(layers), dtype=torch.float32)