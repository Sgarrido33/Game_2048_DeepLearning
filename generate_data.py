import numpy as np
import py7zr
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
from game_2048 import Game2048

ACTION_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}

SNAKE_MATRIX = np.array([
    [2**15, 2**14, 2**13, 2**12],
    [2**8,  2**9,  2**10, 2**11],
    [2**7,  2**6,  2**5,  2**4],
    [2**0,  2**1,  2**2,  2**3]
], dtype=np.float64)

def board_to_one_hot(board):
    one_hot = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val > 0:
                idx = int(np.log2(val))
                if idx < 16:
                    one_hot[idx, i, j] = 1.0
            else:
                one_hot[0, i, j] = 1.0
    return one_hot

def evaluate_board(board):
    empty_cells = np.count_nonzero(board == 0)
    return np.sum(board * SNAKE_MATRIX) + (empty_cells * 100000)

def expectimax_max_node(board, depth):
    best_val = -1
    moved_any = False
    
    for action in ["up", "down", "left", "right"]:
        dummy = Game2048(size=4)
        dummy.board = board.copy()
        moved, _ = dummy._apply_move(action)
        
        if moved:
            moved_any = True
            val = expectimax_chance_node(dummy.board, depth - 1)
            if val > best_val:
                best_val = val
                
    if not moved_any:
        return evaluate_board(board)
        
    return best_val

def expectimax_chance_node(board, depth):
    if depth == 0:
        return evaluate_board(board)
        
    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        return evaluate_board(board)
        
    np.random.shuffle(empties)
    sample = empties[:3]
    
    expected_value = 0
    for r, c in sample:
        for v, prob in [(2, 0.9), (4, 0.1)]:
            tmp_board = board.copy()
            tmp_board[r, c] = v
            expected_value += prob * expectimax_max_node(tmp_board, depth)
            
    return expected_value / len(sample)

def expert_action_deep(game):
    best_score = -1
    best_action = "up"
    legal = game.legal_actions()
    
    if not legal:
        return best_action

    for action in legal:
        dummy = Game2048(size=4)
        dummy.board = game.board.copy()
        moved, _ = dummy._apply_move(action)
        
        if moved:
            score = expectimax_chance_node(dummy.board, depth=1)
            if score > best_score:
                best_score = score
                best_action = action
                
    return best_action

def play_games(episodes):
    X_local, y_local = [], []
    for _ in range(episodes):
        game = Game2048(size=4)
        while True:
            legal = game.legal_actions()
            if not legal:
                break
            
            state_one_hot = board_to_one_hot(game.board)
            action = expert_action_deep(game) 
            
            X_local.append(state_one_hot)
            y_local.append(ACTION_MAP[action])
            
            result = game.step(action)
            if result.done:
                break
    return X_local, y_local

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--cores', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default='./data_raw')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    episodes_per_core = [args.episodes // args.cores] * args.cores
    episodes_per_core[0] += args.episodes % args.cores

    print(f"Generando {args.episodes} partidas")
    
    X_data, y_data = [], []
    
    with mp.Pool(processes=args.cores) as pool:
        results = list(tqdm(pool.imap(play_games, episodes_per_core), total=args.cores))
        
    for X_local, y_local in results:
        X_data.extend(X_local)
        y_data.extend(y_local)

    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int64)

    np.save(os.path.join(args.out_dir, 'X.npy'), X_data)
    np.save(os.path.join(args.out_dir, 'y.npy'), y_data)
    
    print(f"Datos guardados: {len(X_data)} muestras generadas.")

    archive_name = "dataset.7z"
    print(f"Comprimiendo en {archive_name}...")
    with py7zr.SevenZipFile(archive_name, 'w') as archive:
        archive.writeall(args.out_dir, 'dataset')
        
    print("Generación finalizada")

if __name__ == '__main__':
    main()
