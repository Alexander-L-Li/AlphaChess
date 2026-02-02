"""
Lichess data generator for supervised chess training.

Processes compressed PGN files from Lichess database and extracts
training examples as (FEN, move_id, result) tuples.

Download Lichess database files from: https://database.lichess.org/
"""

import zstandard as zstd
import chess.pgn
import io
import torch
import gc
from tqdm import tqdm
from pathlib import Path
from utils import ActionMapper

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

INPUT_FILE = "lichess_db_standard_rated_2025-12.pgn.zst"
OUTPUT_FILE = "lichess_dataset_5M.pt"
TARGET_POSITIONS = 5_000_000
MIN_ELO = 2000
CHUNK_SIZE = 500_000  # Save in chunks to manage memory

# =============================================================================


def parse_result(result_header):
    """Parse PGN result header to numeric value."""
    if result_header == "1-0":
        return 1.0
    elif result_header == "0-1":
        return -1.0
    elif result_header == "1/2-1/2":
        return 0.0
    return None


def process_game(game, mapper, min_elo):
    """
    Process a single game and extract training examples.

    Returns:
        List of (fen, move_id, result) tuples, or empty list if game filtered out
    """
    # Filter by Elo
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if white_elo < min_elo or black_elo < min_elo:
            return []
    except (ValueError, TypeError):
        return []

    # Parse result
    result = parse_result(game.headers.get("Result", "*"))
    if result is None:
        return []

    # Extract positions and moves
    examples = []
    board = game.board()

    for move in game.mainline_moves():
        fen = board.fen()
        move_id = mapper.encode(move)

        if move_id is not None:
            # Store result from perspective of player to move
            player_result = result if board.turn == chess.WHITE else -result
            examples.append((fen, move_id, player_result))

        board.push(move)

    return examples


def save_chunk(data, chunk_id, output_dir):
    """Save a chunk of data to disk."""
    chunk_path = output_dir / f"chunk_{chunk_id:04d}.pt"
    torch.save(data, chunk_path)
    return chunk_path


def merge_chunks(output_dir, final_path, delete_chunks=True):
    """Merge all chunks into a single file."""
    chunk_files = sorted(output_dir.glob("chunk_*.pt"))

    print(f"Merging {len(chunk_files)} chunks...")
    all_data = []

    for chunk_file in tqdm(chunk_files, desc="Merging"):
        chunk_data = torch.load(chunk_file, weights_only=False)
        all_data.extend(chunk_data)

        if delete_chunks:
            chunk_file.unlink()

    print(f"Saving merged dataset ({len(all_data):,} examples)...")
    torch.save(all_data, final_path)

    return len(all_data)


def process_lichess_data():
    """Process Lichess PGN file and extract training positions."""
    print(f"Processing {INPUT_FILE}")
    print(f"Target: {TARGET_POSITIONS:,} positions")
    print(f"Min Elo: {MIN_ELO}")
    print()

    mapper = ActionMapper()

    # Create temp directory for chunks
    output_path = Path(OUTPUT_FILE)
    chunk_dir = output_path.parent / f".chunks_{output_path.stem}"
    chunk_dir.mkdir(exist_ok=True)

    # Statistics
    games_processed = 0
    games_filtered = 0
    positions_collected = 0
    chunk_id = 0
    current_chunk = []

    # Progress bar for positions
    pbar = tqdm(total=TARGET_POSITIONS, desc="Positions", unit="pos")

    with open(INPUT_FILE, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

        while positions_collected < TARGET_POSITIONS:
            try:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    print("\nReached end of file")
                    break

                examples = process_game(game, mapper, MIN_ELO)

                if not examples:
                    games_filtered += 1
                    continue

                games_processed += 1

                # Add examples to current chunk
                for ex in examples:
                    if positions_collected >= TARGET_POSITIONS:
                        break
                    current_chunk.append(ex)
                    positions_collected += 1
                    pbar.update(1)

                # Save chunk if full
                if len(current_chunk) >= CHUNK_SIZE:
                    save_chunk(current_chunk, chunk_id, chunk_dir)
                    chunk_id += 1
                    current_chunk = []
                    gc.collect()

            except Exception:
                # Skip problematic games
                continue

    pbar.close()

    # Save remaining data
    if current_chunk:
        save_chunk(current_chunk, chunk_id, chunk_dir)
        chunk_id += 1

    print()
    print(f"Games processed: {games_processed:,}")
    print(f"Games filtered: {games_filtered:,}")
    print(f"Positions collected: {positions_collected:,}")
    print(f"Chunks created: {chunk_id}")
    print()

    # Merge chunks into final file
    final_count = merge_chunks(chunk_dir, output_path, delete_chunks=True)

    # Clean up chunk directory
    try:
        chunk_dir.rmdir()
    except OSError:
        pass

    print()
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Total examples: {final_count:,}")

    return final_count


if __name__ == "__main__":
    process_lichess_data()
