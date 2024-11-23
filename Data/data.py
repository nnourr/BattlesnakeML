from collections import defaultdict
import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib


class Data:
  def __init__(self, data_root):
    self.action_map = {'ahead': 0, 'turn_left': 1, 'turn_right': 2}
    self.direction_vectors = {
      'up': (0, 1),
      'down': (0, -1),
      'left': (-1, 0),
      'right': (1, 0)
    }
    self.data_root = data_root
    
    self.frames = self.__read_frames()
    self.preprocessed = self.preprocess_frames()
    self.moves = self.__extract_moves()
    
    self.flattened_frames = self.__flatten_frames()
    self.encoded_moves = self.__encode_moves()
    
    self.lda_2 = self.n_lda(2)
    # self.lda_3 = self.n_lda(3)
  
  def lda_transform(self, X):
    X_scaled = self.scaler.transform(X)
    return self.lda.transform(X_scaled)
  
  def n_lda(self, n):
    X = self.flattened_frames
    y = self.encoded_moves
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    self.scaler = scaler
    lda = LinearDiscriminantAnalysis(n_components=n)
    self.lda = lda
    lda_out = lda.fit_transform(X_scaled, y)
    return lda_out
  
  def __read_frames(self):
    frames = []
    for dirpath, dirnames, filenames in os.walk(self.data_root):
        if "frames.json" in filenames:
            frames.extend(self.__read_frames_file(os.path.join(dirpath, "frames.json")))
    return frames
  
  def __read_frames_file(self, file_path):
    with open(file_path, 'r') as _file:
      frames = json.load(_file)
    
    return frames
  
  def preprocess_frames(self, author='coreyja', frames = None):
    preprocessed = []

    if frames == None: frames = self.frames
    self.__new_indexes = [0]

    for i, game_state in enumerate(frames):
        turn = game_state["Turn"]
        food = game_state["Food"]
        food = [(f["X"], f["Y"]) for f in food]
        snakes = game_state["Snakes"]

        # Initialize lists for player and enemy snakes
        player_body = None
        enemy_body = None
        enemy_score = None
        player_health = None
        player_score = None
        enemy_health = False
        end_frame = False

        scores = self.score_game_board(game_state)

        for snake in snakes:
            if (snake['Death'] != None):
               end_frame = True
            snake_body = [(s["X"], s["Y"]) for s in snake["Body"]]
            if snake["Author"] == author:
                player_body = snake_body
                player_health = snake["Health"]
                player_score = scores[snake["ID"]]
            else:
                enemy_body = snake_body
                enemy_health = snake["Health"]
                enemy_score = scores[snake["ID"]]

        if player_body is None:
            continue  # Skip if the player's snake is not found

        if end_frame:
            self.__new_indexes.append(len(preprocessed)+1)

        # Calculate the distance between the player's snake head and the enemy's snake head
        player_head = player_body[0]
        enemy_head = enemy_body[0]

        opp_distance = self.calculate_snake_distance(player_head, enemy_body)
        player_distance = self.calculate_snake_distance(enemy_head, player_body)
        pairwise_distance = self.calculate_pairwise_distance(enemy_body, player_body)
        wall_distance = self.calculate_wall_distance(player_body)
        food_distance = self.calculate_food_distance(player_body, food)
        valid_spaces = self.calculate_valid_moves(player_body, enemy_body)

        # Flatten features into a vector (simple representation)
        features = {
            "turn": turn,
            "food_positions": food,
            "player_body": player_body,
            "enemy_body": enemy_body,
            "health": player_health,
            "opp_health": enemy_health,
            "opp_distance": opp_distance,
            "player_distance": player_distance,
            "pairwise_distance": pairwise_distance,
            "food_distance": food_distance,
            "wall_distance": wall_distance,
            "valid_spaces": valid_spaces,
            "player_score": player_score,
            "enemy_score": enemy_score
        }
        preprocessed.append(features)

    return preprocessed
  
  def __extract_moves(self):
    moves = []
    end_frames = [index - 1 for index in self.__new_indexes[1:]]  # Get the last frame of each game except the final game
    
    # Need at least two body segments to determine direction
    for i, turn_data in enumerate(self.preprocessed):
      # Skip if this is an end frame
      if i in end_frames:
        continue
            
      player_snake = turn_data['player_body']
      if len(player_snake) < 2:
        continue
            
      head = player_snake[0]
      neck = player_snake[1]
        
      # Get next frame's head position if available
      if i + 1 < len(self.preprocessed) and i + 1 not in self.__new_indexes:
        next_head = self.preprocessed[i + 1]['player_body'][0]
            
        # For first frame of each game, determine initial direction from first move
        if i in self.__new_indexes:
          # Calculate movement vector
          dx = next_head[0] - head[0]
          dy = next_head[1] - head[1]
                
          # First determine which direction the snake actually moved
          next_direction = None
          for direction, vector in self.direction_vectors.items():
              if vector == (dx, dy):
                  next_direction = direction
                  break
                
          # Set initial direction based on the move:
          # If snake moves right -> was facing right
          # If snake moves left -> was facing left
          # If snake moves up -> was facing up
          # If snake moves down -> was facing down
          current_direction = next_direction if next_direction else 'right'
        else:
          # Normal case - determine current and next directions
          current_direction = self.__get_current_direction(head, neck)
          next_direction = self.__get_current_direction(next_head, head)
            
        if current_direction and next_direction:
          relative_move = self.__get_relative_move(current_direction, next_direction)
          moves.append({
            "Turn": i,
            "Move": relative_move
          })

    # Remove end frames from preprocessed data
    for i, index in enumerate(end_frames):
      self.preprocessed.pop(index - i)
    
    return moves
  
  def __get_current_direction(self, head, neck):
    """Determine current direction based on head and neck positions"""
    dx = head[0] - neck[0]
    dy = head[1] - neck[1]
    for direction, vector in self.direction_vectors.items():
      if vector == (dx, dy):
        return direction
    return None

  def __get_relative_move(self, current_direction, next_direction):
    """Convert absolute direction change to relative move"""
    if current_direction == next_direction:
      return 'ahead'
        
    # Define what directions are to the left/right of each direction
    relative_directions = {
      'up': {'left': 'left', 'right': 'right'},
      'right': {'left': 'up', 'right': 'down'},
      'down': {'left': 'right', 'right': 'left'},
      'left': {'left': 'down', 'right': 'up'}
    }
    
    if next_direction == relative_directions[current_direction]['left']:
      return 'turn_left'
    elif next_direction == relative_directions[current_direction]['right']:
      return 'turn_right'
    return 'ahead'  # Default to ahead if something unexpected happens
  
  def __encode_moves(self):
    return np.array([self.action_map[move['Move']] for move in self.moves])
  
  def decode_moves(self, encoded_moves):
    decoded = list(self.action_map.keys())
    return [decoded[move] for move in encoded_moves]
  
  def __flatten_frames(self):
    flattened_frames = []
    for i, frame in enumerate(self.preprocessed):
      flattened_frames.append(self.flatten_frame_i_to_list(frame))
      
    return np.array(flattened_frames)
  
  def flatten_frame_i_to_list(self, frame):

    # add length to arrays
    return np.append(self.flatten_frame_to_list(frame), [len(frame["player_body"]), len(frame["enemy_body"]), len(frame["food_positions"]), frame["health"], frame['player_score'], frame['enemy_score'], *self.flatten_frame_to_board(frame)], axis=0)
  
  def anti_symmetric_pad(self, array, target_size):
    """
    Pads a 1D array to the specified target size using anti-symmetric padding.

    Args:
        array (numpy.ndarray): The input 1D array.
        target_size (int): The desired size of the padded array.

    Returns:
        numpy.ndarray: The anti-symmetrically padded array.
    """
    current_size = len(array)
    
    if current_size >= target_size:
      # Truncate if the array is larger than or equal to the target size
      return array[:target_size]
    
    # Calculate the total amount of padding needed
    total_pad = target_size - current_size
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad

    # Handle edge cases for padding
    left = -np.flip(array[:min(left_pad, current_size)])
    right = -np.flip(array[-min(right_pad, current_size):])

    # Adjust padding if the array is smaller than the required padding
    while len(left) < left_pad:
      left = np.concatenate((left, -np.flip(array[:min(left_pad - len(left), current_size)])))
    while len(right) < right_pad:
      right = np.concatenate((right, -np.flip(array[-min(right_pad - len(right), current_size):])))

    # Truncate any extra padding to ensure exact match
    left = left[-left_pad:]
    right = right[:right_pad]

    # Concatenate the padded parts with the original array
    padded_array = np.concatenate((left, array, right))
    
    return padded_array
  
  def flatten_frame_to_list(self, frame):
    food_positions = [coord for f in frame["food_positions"] for coord in f]
    player_body = [coord for s in frame["player_body"] for coord in s]
    enemy_bodies = [coord for b in frame["enemy_body"] for coord in b]
    wall_distance = [dist for b in frame["wall_distance"] for dist in b]
    food_distance = [dist for b in frame["food_distance"] for dist in b]
    opp_distance = frame["opp_distance"]
    player_distance = frame["player_distance"]
    pairwise_distance = frame["pairwise_distance"]
    valid_spaces = frame["valid_spaces"]

     # Define the desired length for padding
    desired_length = 100

    # Function to pad arrays to the desired length
    def pad_to_length(arr, length):
      # return np.pad(arr, (0, max(0, length - len(arr))), mode='constant')
      return self.anti_symmetric_pad(arr, length)

    # Pad each feature array to the desired length
    # turn_padded = pad_to_length(turn, desired_length)
    food_positions_padded = pad_to_length(food_positions, desired_length)
    player_body_padded = pad_to_length(player_body, desired_length)
    enemy_bodies_padded = pad_to_length(enemy_bodies, desired_length)

    opp_distance_padded = pad_to_length(opp_distance, desired_length)
    player_distance_padded = pad_to_length(player_distance, desired_length)
    pairwise_distance_padded = pad_to_length(pairwise_distance, 2000)
    wall_distance_padded = pad_to_length(wall_distance, 400)
    food_distance_padded = pad_to_length(food_distance, 400)

    # Concatenate all arrays
    return np.concatenate([food_positions_padded, player_body_padded, enemy_bodies_padded, player_distance_padded, opp_distance_padded, pairwise_distance_padded, wall_distance_padded, food_distance_padded, valid_spaces])
  
  def flatten_frame_to_board(self, frame):
    # Convert the board into a flattened array
    board_size = 11  # for an 11x11 board
    grid = np.zeros((board_size, board_size))

    # Mark snake body positions as -1
    for segment in frame['player_body']:
      if segment[0] >= 11 or segment[1] >= 11: continue
      grid[segment[0], segment[1]] = -1
    
    # Mark opponent snakes
    for segment in frame['enemy_body']:
      if segment[0] >= 11 or segment[1] >= 11: continue
      grid[segment[0], segment[1]] = -2

    # Mark food position as 1
    for food in frame['food_positions']:
      grid[food[0], food[1]] = 1

    # Flatten the grid
    return grid.flatten()

  def __flatten_frame_i_to_board(self, frame, i):
    # Convert the board into a flattened array
    board_size = 11  # for an 11x11 board
    grid = np.zeros((board_size, board_size))

    # Mark snake body positions as -1
    for segment in frame['player_body']:
      grid[segment[0], segment[1]] = -1
    
    # Mark opponent snakes
    for segment in frame['enemy_body']:
      grid[segment[0], segment[1]] = -2

    # Mark food position as 1
    for food in frame['food_positions']:
      grid[food[0], food[1]] = 1

    # Flatten the grid
    return np.append(grid.flatten(), i)

  def __flatten_frame_len_to_board(self, frame, i):
    # Convert the board into a flattened array
    board_size = 11  # for an 11x11 board
    grid = np.zeros((board_size, board_size))

    # Mark snake body positions as -1
    for segment in frame['player_body']:
      grid[segment[0], segment[1]] = -1
    
    # Mark opponent snakes
    for segment in frame['enemy_body']:
      grid[segment[0], segment[1]] = -2

    # Mark food position as 1
    for food in frame['food_positions']:
      grid[food[0], food[1]] = 1

    # Flatten the grid
    return np.append(grid.flatten(), [len(frame['player_body']), len(frame['enemy_body'])*2, len(frame['food_positions'])*-1], axis=0)
  
  def calculate_euclidean_distance(self):
    pairwise_distances = pdist(self.flattened_frames, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, 0)  # Ensuring D(a, a) = 0
    return distance_matrix
  
  def multi_dim_scaling(self, euclidean_data, n_components=2):
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    return mds.fit_transform(euclidean_data)
  
  def calculate_snake_distance(self, head, body):
    return np.array([np.sum(np.abs(np.array(head) - np.array(segment))) for segment in body])

  def calculate_pairwise_distance(self, body1, body2):
    return cdist(body1, body2, 'cityblock').ravel()
  
  def calculate_wall_distance(self, body, board_size=11):
    # Distance from upper
      # 11 - y
    # Distance from lower
      # y
    # Distance from left
      # x
    # Distance from right
      # 11 - x

    distances = []

    for segment in body:
      x, y = segment
      distance_to_top = board_size - y
      distance_to_right = board_size - x
      distances.append([distance_to_top, distance_to_right])

    return np.array(distances)
  
  def calculate_food_distance(self, body, foods):
    return np.array([[np.sum(np.abs(np.array(segment) - np.array(food))) for food in foods] for segment in body])
  
  def calculate_valid_moves(self, player, opp, board_size=11):
    """Calculate valid moves relative to snake's current direction.
    Returns a mask for [ahead, turn_left, turn_right]
    """
    if len(player) < 2:
      return [1, 1, 1]  # Allow all moves if we don't have enough info
        
    head = player[0]
    neck = player[1]
    
    # Get current direction
    current_direction = self.__get_current_direction(head, neck)
    if not current_direction:
      return [1, 1, 1]  # Allow all moves if direction is invalid
        
    # Define what coordinates would be reached by each relative move
    relative_coords = {
      'up': {
        'ahead': (head[0], head[1] + 1),
        'turn_left': (head[0] - 1, head[1]),
        'turn_right': (head[0] + 1, head[1])
      },
      'right': {
        'ahead': (head[0] + 1, head[1]),
        'turn_left': (head[0], head[1] + 1),
        'turn_right': (head[0], head[1] - 1)
      },
      'down': {
        'ahead': (head[0], head[1] - 1),
        'turn_left': (head[0] + 1, head[1]),
        'turn_right': (head[0] - 1, head[1])
      },
      'left': {
        'ahead': (head[0] - 1, head[1]),
        'turn_left': (head[0], head[1] - 1),
        'turn_right': (head[0], head[1] + 1)
      }
    }
    
    # Get coordinates for each possible move based on current direction
    possible_moves = relative_coords[current_direction]
    
    # Initialize mask for [ahead, turn_left, turn_right]
    mask = [1, 1, 1]
    
    # Get opponent head for potential collision detection
    opp_head = opp[0]
    
    # Calculate potential moves for the opponent's head
    opponent_danger_zone = [
      (opp_head[0], opp_head[1] + 1),  # Up
      (opp_head[0], opp_head[1] - 1),  # Down
      (opp_head[0] - 1, opp_head[1]),  # Left
      (opp_head[0] + 1, opp_head[1])   # Right
    ]
    
    # Filter out moves that are outside the board
    opponent_danger_zone = [
      move for move in opponent_danger_zone 
      if 0 <= move[0] < board_size and 0 <= move[1] < board_size
    ]
    
    # Check if moves would result in collision with opponent's potential moves
    player_next_states = [possible_moves['ahead'], possible_moves['turn_left'], possible_moves['turn_right']]
    avoid_head = bool(set(opponent_danger_zone) & set(player_next_states))
    
    # Check each relative move
    for i, (move_type, coords) in enumerate([
      ('ahead', possible_moves['ahead']),
      ('turn_left', possible_moves['turn_left']),
      ('turn_right', possible_moves['turn_right'])
    ]):
      x, y = coords
      # Check if move is invalid (wall collision, body collision, or dangerous head-to-head)
      if (x < 0 or x >= board_size or  # Wall collision
        y < 0 or y >= board_size or  # Wall collision
        (x, y) in player or          # Self collision
        (x, y) in opp or             # Opponent body collision
        ((x, y) in opponent_danger_zone and avoid_head)):  # Dangerous head-to-head
        mask[i] = 0
    
    return mask
  
  def score_game_board(self, game_state: dict) -> dict:
    '''
    parameter:
        game_state: full json dict of game data

    returns dict of snake_id to float score
    '''
    # Get scores from all scoring methods
    snake_flood_score = self.get_flood_score(game_state)
    snake_length_score = self.get_length_score(game_state)
    snake_area_score = self.get_area_control_score(game_state)

    # Combines scores together with weights
    snake_to_score = defaultdict(float)
    
    # Weight factors for different scoring components
    FLOOD_WEIGHT = 0.3
    LENGTH_WEIGHT = 0.3
    AREA_WEIGHT = 0.4  # Giving slightly more weight to area control
    
    for k, v in snake_flood_score.items():
        snake_to_score[k] += v * FLOOD_WEIGHT
    
    for k, v in snake_length_score.items():
        snake_to_score[k] += v * LENGTH_WEIGHT
        
    for k, v in snake_area_score.items():
        snake_to_score[k] += v * AREA_WEIGHT

    return snake_to_score

  def get_flood_score(self, game_state: dict) -> dict:
      '''
      parameter:
          game state: full json dict of game data

      returns map of snake IDs to a float score (based on floodfill)
      '''

      # Flood Fill Algo?
      W = 11
      H = W

      board = [[""] * H] * W

      q = list()

      def is_valid(x, y):
          if x < 0 or W <= x:
              return False
          if y < 0 or H <= y:
              return False
          if board[x][y] != '':
              return False
          return True
      
      for snake in game_state['Snakes']:
          # Adds head to queue
          q.append(snake['Body'][0] | {'ID': snake['ID']})

          # Marks area where snake sits as controlled
          for b in snake['Body']:
              if (not is_valid(b['X'], b['Y'])): continue
              board[b['X']][b['Y']] = snake['ID']

      # Loops through snake moves until board filled
      while q:
          cur_coord = q.pop(0)
          x = cur_coord['X']
          y = cur_coord['Y']
          id = cur_coord['ID']

          if (not is_valid(x, y)): continue
          board[x][y] = id

          if is_valid(x + 1, y):
              q.append({'ID': id, 'X': x + 1, 'Y': y})

          if is_valid(x - 1, y):
              q.append({'ID': id, 'X': x - 1, 'Y': y})

          if is_valid(x, y + 1):
              q.append({'ID': id, 'X': x, 'Y': y + 1})

          if is_valid(x, y - 1):
              q.append({'ID': id, 'X': x, 'Y': y - 1})

      snake_to_score = defaultdict(float)
      for i in board:
          for j in i:
              if j == '':
                  continue
              snake_to_score[j] += 1

      total = sum(snake_to_score.values())
      for i in snake_to_score:
          snake_to_score[i] /= total

      return snake_to_score


  def get_length_score(self, game_state: dict) -> dict:
      '''
      parameter:
          game state: full json dict of game data

      returns map of snake IDs to a float score (based on length)
      '''

      snake_to_score = defaultdict(float)

      for snake in game_state['Snakes']:
          snake_to_score[snake['ID']] = len(snake['Body'])

      total = sum(snake_to_score.values())
      for i in snake_to_score:
          snake_to_score[i] /= total

      return snake_to_score

  def get_area_control_score(self, game_state: dict) -> dict:
    """
    Calculates area control score for each snake based on the "shadow" concept.
    A snake controls the area that extends from its body away from the opponent's body.
    
    Args:
        game_state: full json dict of game data
    
    Returns:
        dict: Map of snake IDs to their area control scores
    """
    board_size = 11
    board = np.zeros((board_size, board_size), dtype=int)
    snake_scores = defaultdict(float)
    
    # Get snakes
    snakes = game_state.get('board', {}).get('snakes', [])
    if len(snakes) != 2:  # We need exactly 2 snakes for this calculation
        return snake_scores
    
    snake1, snake2 = snakes[0], snakes[1]
    
    # Mark snake bodies on the board
    # 1 for snake1, 2 for snake2
    for i, snake in enumerate([snake1, snake2], 1):
        for segment in snake['body']:
            x, y = segment['x'], segment['y']
            if 0 <= x < board_size and 0 <= y < board_size:
                board[y, x] = i
    
    # Calculate area control for each snake
    for y in range(board_size):
        for x in range(board_size):
            if board[y, x] == 0:  # Empty space
                # Calculate distances to nearest body segments of each snake
                min_dist_snake1 = float('inf')
                min_dist_snake2 = float('inf')
                
                # Check distance to snake1's body
                for segment in snake1['body']:
                    dist = abs(x - segment['x']) + abs(y - segment['y'])  # Manhattan distance
                    min_dist_snake1 = min(min_dist_snake1, dist)
                
                # Check distance to snake2's body
                for segment in snake2['body']:
                    dist = abs(x - segment['x']) + abs(y - segment['y'])
                    min_dist_snake2 = min(min_dist_snake2, dist)
                
                # Assign control based on relative distances
                if min_dist_snake1 < min_dist_snake2:
                    snake_scores[snake1['id']] += 1
                elif min_dist_snake2 < min_dist_snake1:
                    snake_scores[snake2['id']] += 1
                # If equal distances, no points awarded
    
    # Normalize scores
    total_score = sum(snake_scores.values())
    if total_score > 0:
        for snake_id in snake_scores:
            snake_scores[snake_id] /= total_score
    
    return snake_scores
