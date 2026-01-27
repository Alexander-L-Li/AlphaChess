import torch
import chess
import numpy as np
from utils import ActionMapper, board_to_tensor
from nn import ChessNet
from mcts import MCTS, predict_masked
import pygame
import os
import sys
from pathlib import Path

# --- Configuration ---
WIDTH, HEIGHT = 600, 600
SQ_SIZE = WIDTH // 8
MAX_FPS = 15
IMAGES = {}

PIECE_IMAGE_DIR = Path(__file__).resolve().parent / "assets" / "pieces"
# Map python-chess piece symbols -> local filenames.
# Convention:
# - White pieces: `P.png`, `N.png`, `B.png`, `R.png`, `Q.png`, `K.png`
# - Black pieces: `BP.png`, `BN.png`, `BB.png`, `BR.png`, `BQ.png`, `BK.png`
#
# python-chess uses lowercase for black pieces: 'p','n','b','r','q','k'.
PIECE_FILE_CANDIDATES: dict[str, list[str]] = {
    "p": ["BP.png"],
    "n": ["BN.png"],
    "b": ["BB.png"],
    "r": ["BR.png"],
    "q": ["BQ.png"],
    "k": ["BK.png"],
    "P": ["P.png"],
    "N": ["N.png"],
    "B": ["B.png"],
    "R": ["R.png"],
    "Q": ["Q.png"],
    "K": ["K.png"],
}

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

def _fallback_piece_surface(piece_char: str) -> pygame.Surface:
    """
    If a local PNG is missing/unloadable, draw a simple piece marker so the UI still runs.
    (Avoids pygame.font dependency.)
    """
    surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    is_white = piece_char.isupper()
    fill = pygame.Color("white") if is_white else pygame.Color("black")
    outline = pygame.Color("black") if is_white else pygame.Color("white")
    cx, cy = SQ_SIZE // 2, SQ_SIZE // 2
    r = SQ_SIZE // 2 - 4
    pygame.draw.circle(surf, fill, (cx, cy), r)
    pygame.draw.circle(surf, outline, (cx, cy), r, width=2)
    return surf


def _load_png_via_pillow(path: Path) -> pygame.Surface:
    """
    Decode PNG via Pillow, then convert to a pygame Surface.
    This works even when pygame was built without PNG support.
    """
    if Image is None:
        raise RuntimeError(
            "Pillow not installed, and this pygame build cannot load PNGs.\n"
            "Fix:\n"
            "  python -m pip install pillow\n"
        )
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    return pygame.image.frombuffer(data, (w, h), "RGBA").convert_alpha()

def load_images():
    """
    Loads piece images from a local folder.

    Expected filenames (PNG) in `assets/pieces/`:
    - `BP.png`, `BN.png`, `BB.png`, `BR.png`, `BQ.png`, `BK.png`
    - `P.png`, `N.png`, `B.png`, `R.png`, `Q.png`, `K.png`
    """
    print("Loading assets...")
    for piece_char, candidates in PIECE_FILE_CANDIDATES.items():
        path: Path | None = None
        for name in candidates:
            p = PIECE_IMAGE_DIR / name
            if p.exists():
                path = p
                break
        try:
            if path is None:
                raise FileNotFoundError(f"No local PNG found for '{piece_char}'. Tried: {candidates}")

            try:
                img = pygame.image.load(str(path)).convert_alpha()
            except pygame.error:
                img = _load_png_via_pillow(path)

            img = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            IMAGES[piece_char] = img
        except Exception as e:
            print(f"Warning: could not load {piece_char} from {path}: {e}")
            IMAGES[piece_char] = _fallback_piece_surface(piece_char)

def draw_board(screen):
    """Draws the checkerboard pattern."""
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(8):
        for c in range(8):
            color = colors[((r + c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    """Draws the pieces on top of the board."""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convert chess square (0-63) to x,y coordinates
            # File (0-7) is x, Rank (0-7) is y. Note: Rank 0 is bottom in chess, but y=7 in pygame.
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            
            piece_char = piece.symbol() # 'P', 'k', etc.
            if piece_char in IMAGES:
                screen.blit(IMAGES[piece_char], pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def highlight_square(screen, square):
    """
    Highlight a selected square with a semi-transparent overlay.

    Args:
        screen: The pygame display surface.
        square: The chess square index to highlight (0-63), or None.
    """
    if square is not None:
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100) # Transparency
        s.fill(pygame.Color('yellow'))
        screen.blit(s, (col*SQ_SIZE, row*SQ_SIZE))

def get_ai_move(model, board, mapper, device):
    """Wrapper to run MCTS and get the best move."""
    print("AI is thinking...")
    # Adjust simulations: 50=Fast/Weak, 200=Decent, 800=Strong(Slow)
    mcts = MCTS(model, device, mapper)
    root = mcts.search(board, num_simulations=800)
    
    best_move = None
    max_visits = -1
    for move, child in root.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_move = move
            
    return best_move

def main():
    """
    Main entry point for the interactive chess game.

    Initializes pygame, loads the trained model, and runs the game loop
    where a human player competes against the AI.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('My Chess Engine (MCTS)')
    clock = pygame.time.Clock()
    
    # 1. Load AI
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mapper = ActionMapper()
    model = ChessNet(action_size=mapper.vocab_size).to(device)
    
    try:
        model.load_state_dict(torch.load("modal_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("No model found. Please run training first!")
        return

    # 2. Setup Game
    load_images()
    board = chess.Board()
    player_is_white = True # Change to False to let AI play White
    
    selected_square = None # Keep track of last click
    running = True
    
    while running:
        human_turn = (board.turn == chess.WHITE and player_is_white) or \
                     (board.turn == chess.BLACK and not player_is_white)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN and human_turn:
                location = pygame.mouse.get_pos() # (x, y)
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                # Convert PyGame row to Chess Rank
                rank = 7 - row
                clicked_sq = chess.square(col, rank)
                
                if selected_square == clicked_sq:
                    selected_square = None # Deselect
                else:
                    if selected_square is not None:
                        # Try to move
                        move = chess.Move(selected_square, clicked_sq)
                        
                        # Auto-promotion to Queen for simplicity in UI
                        if board.piece_at(selected_square).piece_type == chess.PAWN:
                            if (board.turn == chess.WHITE and rank == 7) or \
                               (board.turn == chess.BLACK and rank == 0):
                                move = chess.Move(selected_square, clicked_sq, promotion=chess.QUEEN)

                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                        else:
                            # If illegal, just select the new piece (if it's ours)
                            piece = board.piece_at(clicked_sq)
                            if piece and piece.color == board.turn:
                                selected_square = clicked_sq
                            else:
                                selected_square = None
                    else:
                        # Select a piece
                        piece = board.piece_at(clicked_sq)
                        if piece and piece.color == board.turn:
                            selected_square = clicked_sq

        # Draw Game State
        draw_board(screen)
        highlight_square(screen, selected_square)
        draw_pieces(screen, board)
        
        # Check Endgame
        if board.is_game_over():
            # Some pygame builds (esp. very new Python versions) may not ship with font support.
            if hasattr(pygame, "font") and getattr(pygame.font, "SysFont", None):
                try:
                    text = pygame.font.SysFont('Arial', 32).render(
                        f"Game Over: {board.result()}",
                        True,
                        pygame.Color('Red')
                    )
                    screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2))
                except Exception:
                    pass
            pygame.display.flip()
            continue

        pygame.display.flip()
        
        # AI Logic (Only if game is running and it's AI turn)
        if not human_turn and not board.is_game_over():
            # Force a UI refresh before AI freezes the thread with thinking
            pygame.event.pump() 
            ai_move = get_ai_move(model, board, mapper, device)
            if ai_move:
                board.push(ai_move)
            else:
                print("AI Resigns")
                running = False

        clock.tick(MAX_FPS)

if __name__ == "__main__":
    main()