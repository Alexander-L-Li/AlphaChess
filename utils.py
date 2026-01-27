import chess
import numpy as np
import torch

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

def board_to_tensor(board):
    """
    Convert a chess.Board to a tensor representation for the neural network.

    Creates a 13-plane tensor encoding the board state:
    - Planes 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    - Planes 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    - Plane 12: Turn indicator (all 1s if white to move, all 0s if black)

    Each piece plane is an 8x8 binary mask where 1 indicates the presence
    of that piece type and color on the corresponding square.

    Args:
        board: The chess.Board position to convert.

    Returns:
        torch.Tensor of shape (13, 8, 8) with float32 dtype.
    """
    # 6 pieces * 2 colors = 12 planes. 
    # +1 plane for turn (all 0 for black, all 1 for white)
    # Shape: (13, 8, 8)
    pieces = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP, 
        chess.ROOK, chess.QUEEN, chess.KING
    ]
    layers = []
    
    # Generate 12 planes for pieces
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in pieces:
            layer = np.zeros((8, 8), dtype=np.float32)
            # Get square set for this piece type and color
            for square in board.pieces(piece_type, color):
                rank, file = chess.square_rank(square), chess.square_file(square)
                layer[rank, file] = 1.0
            layers.append(layer)
            
    # Generate 1 plane for turn
    turn_layer = np.zeros((8, 8), dtype=np.float32)
    if board.turn == chess.WHITE:
        turn_layer.fill(1.0)
    layers.append(turn_layer)

    return torch.tensor(np.stack(layers), dtype=torch.float32)