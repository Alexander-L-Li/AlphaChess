import zstandard as zstd
import chess.pgn
import io
import torch
import numpy as np
from tqdm import tqdm
from utils import ActionMapper

mapper = ActionMapper()

def process_lichess_data(file_path, max_games=10000, elo_threshold=2000):
    """
    Reads a .zst PGN file and saves a list of (FEN, move_index, result).
    """
    print(f"Processing {file_path}...")
    data = []
    
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        
        count = 0
        pbar = tqdm(total=max_games)
        
        while count < max_games:
            game = chess.pgn.read_game(text_stream)
            if game is None: break # End of file
            
            # Filter: Quality Control (Learn only from good players)
            # You might need to handle "?" Elo strings gracefully here
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                if white_elo < elo_threshold or black_elo < elo_threshold:
                    continue
            except ValueError:
                continue

            # Get Result (1=White Win, 0=Black Win, 0.5=Draw)
            res_header = game.headers.get("Result", "*")
            if res_header == "1-0": game_result = 1.0
            elif res_header == "0-1": game_result = -1.0
            elif res_header == "1/2-1/2": game_result = 0.0
            else: continue # Skip unfinished games
            
            board = game.board()
            for move in game.mainline_moves():
                # 1. Store state (FEN is compact)
                fen = board.fen()
                
                # 2. Store action (Mapped ID)
                # We want to learn: In position 'fen', the expert played 'move'
                move_id = mapper.encode(move)
                
                if move_id is not None:
                    data.append((fen, move_id, game_result))
                
                board.push(move)
                
            count += 1
            pbar.update(1)
            
        pbar.close()

    print(f"Processed {len(data)} moves from {count} games.")
    print("Saving to disk...")
    torch.save(data, "lichess_dataset_10k.pt")
    print("Done!")

process_lichess_data("/Users/alxli/Documents/lichess_db_standard_rated_2025-12.pgn.zst")