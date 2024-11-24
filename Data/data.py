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
    self.action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    self.data_root = data_root
    
    self.frames = self.__read_frames()
    self.preprocessed = self.preprocess_frames()
    self.moves = self.__extract_moves()
    
    self.flattened_frames = self.__flatten_frames()
    self.encoded_moves = self.__encode_moves()
    
    # self.lda_2 = self.n_lda(2)
    self.lda_3 = self.n_lda(3)
  
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

        scores, control_map = self.score_game_board(game_state)

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
            self.__new_indexes.append(len(preprocessed))

        # Calculate the distance between the player's snake head and the enemy's snake head
        player_head = player_body[0]
        enemy_head = enemy_body[0]

        opp_distance = self.calculate_snake_distance(player_head, enemy_body)
        player_distance = self.calculate_snake_distance(enemy_head, player_body)
        pairwise_distance = self.calculate_pairwise_distance(enemy_body, player_body)
        player_wall_distance = self.calculate_wall_distance(player_body)
        opp_wall_distance = self.calculate_wall_distance(enemy_body)
        player_food_distance = self.calculate_food_distance(player_body, food)
        opp_food_distance = self.calculate_food_distance(enemy_body, food)
        player_valid_spaces = self.calculate_valid_moves(player_body, enemy_body)
        enemy_valid_spaces = self.calculate_valid_moves(enemy_body, player_body)

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
            "player_food_distance": player_food_distance,
            "player_wall_distance": player_wall_distance,
            "enemy_food_distance": opp_food_distance,
            "enemy_wall_distance": opp_wall_distance,
            "player_valid_spaces": player_valid_spaces,
            "enemy_valid_spaces": enemy_valid_spaces,
            "player_score": player_score,
            "enemy_score": enemy_score,
            "control_map": control_map,
        }
        preprocessed.append(features)

    return preprocessed
  
  def __extract_moves(self):
    moves = []

    # Initialize previous head positions for the given snakeID
    previous_position = ''

    for i, turn_data in enumerate(self.preprocessed):
        player_snake = turn_data['player_body']

        if (i in self.__new_indexes):
           previous_position = ''

        # Get the current snake's head position
        head_position = player_snake[0]

        if previous_position == '':
            # Initialize with the first turn's position
            previous_position = head_position
            continue

        # Determine the move based on the change in position
        current_position = head_position

        move = None
        if current_position[0] > previous_position[0]:
            move = "right"
        elif current_position[0] < previous_position[0]:
            move = "left"
        elif current_position[1] < previous_position[1]:
            move = "down"
        elif current_position[1] > previous_position[1]:
            move = "up"

        # Store the move along with the turn number
        moves.append({
            "Turn": i,
            "Move": move
        })

        # Update previous position
        previous_position = current_position

    for i, index in enumerate(self.__new_indexes):
        self.preprocessed.pop(index - (i + 1))
    return moves
  
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
    return np.append(self.flatten_frame_to_list(frame), [len(frame["player_body"]), len(frame["enemy_body"]), len(frame["food_positions"]), frame["opp_health"], frame["health"], frame['player_score'], frame['enemy_score'], *self.flatten_frame_to_board(frame)], axis=0)
  
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
    player_wall_distance = [dist for b in frame["player_wall_distance"] for dist in b]
    enemy_wall_distance = [dist for b in frame["enemy_wall_distance"] for dist in b]
    player_food_distance = [dist for b in frame["player_food_distance"] for dist in b]
    enemy_food_distance = [dist for b in frame["enemy_food_distance"] for dist in b]
    control_map = [score for b in frame["control_map"] for score in b]
    opp_distance = frame["opp_distance"]
    player_distance = frame["player_distance"]
    pairwise_distance = frame["pairwise_distance"]
    player_valid_spaces = frame["player_valid_spaces"]
    enemy_valid_spaces = frame["enemy_valid_spaces"]
    player_score = frame["player_score"]
    enemy_score = frame["enemy_score"]

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
    player_wall_distance_padded = pad_to_length(player_wall_distance, 400)
    enemy_wall_distance_padded = pad_to_length(enemy_wall_distance, 400)
    player_food_distance_padded = pad_to_length(player_food_distance, 400)
    enemy_food_distance_padded = pad_to_length(enemy_food_distance, 400)

    # Concatenate all arrays
    return np.concatenate([food_positions_padded,
     player_body_padded,
     enemy_bodies_padded,
     player_distance_padded,
     opp_distance_padded,
     pairwise_distance_padded,
     player_wall_distance_padded,
     enemy_wall_distance_padded,
     player_food_distance_padded,
     enemy_food_distance_padded,
     player_valid_spaces,
     enemy_valid_spaces,
     control_map])
  
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
  
  def multi_dim_scaling(self, data, dim = 2):
    mds = MDS(n_components=dim, random_state=0, dissimilarity='precomputed')  
    return mds.fit_transform(data)
  
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
    mask = [1, 1, 1, 1]  # Initialize mask for [up, down, left, right]

    head_x, head_y = player[0]  # Get the head position of the player's snake

    opp_x, opp_y = opp[0]  # Get the head position of the opponent snake

    # Calculate potential moves for the opponent's head
    opponent_danger_zone = [
        (opp_x, opp_y + 1),  # Up
        (opp_x, opp_y - 1),  # Down
        (opp_x - 1, opp_y),  # Left
        (opp_x + 1, opp_y)   # Right
    ]

    # Filter out moves that are outside the board
    opponent_danger_zone = [
        move for move in opponent_danger_zone 
        if 0 <= move[0] < board_size and 0 <= move[1] < board_size
    ]
    
    player_next_states = [
        (head_x, head_y + 1),  # Up
        (head_x, head_y - 1),  # Down
        (head_x - 1, head_y),  # Left
        (head_x + 1, head_y)   # Right
    ]
    
    avoid_head = bool(set(opponent_danger_zone) & set(player_next_states))
    

    # Check up move
    if head_y + 1 >= board_size or (head_x, head_y + 1) in player or (head_x, head_y + 1) in opp or ((head_x, head_y + 1) in opponent_danger_zone and avoid_head):
        mask[0] = 0

    # Check down move
    if head_y - 1 < 0 or (head_x, head_y - 1) in player or (head_x, head_y - 1) in opp or ((head_x, head_y - 1) in opponent_danger_zone and avoid_head):
        mask[1] = 0

    # Check left move
    if head_x - 1 < 0 or (head_x - 1, head_y) in player or (head_x - 1, head_y) in opp or ((head_x - 1, head_y) in opponent_danger_zone and avoid_head):
        mask[2] = 0

    # Check right move
    if head_x + 1 >= board_size or (head_x + 1, head_y) in player or (head_x + 1, head_y) in opp or ((head_x + 1, head_y) in opponent_danger_zone and avoid_head):
        mask[3] = 0

    return mask
  
  def score_game_board(self, game_state: dict) -> dict:
    '''
    parameter:
        game_state: full json dict of game data

    returns dict of snake_id to float score
    '''

    # Gets scores from both algorithms
    snake_flood_score = self.get_flood_score(game_state)
    snake_length_score = self.get_length_score(game_state)
    area_scores, wall_control_board, directional_control_board, area_control_board = self.get_area_control(game_state)

    # combine boards
    combined_board = wall_control_board + directional_control_board + area_control_board

    # Combines scores together
    snake_to_score = defaultdict(float)

    for k, v in snake_flood_score.items():
        snake_to_score[k] += v

    for k, v in snake_length_score.items():
        snake_to_score[k] += v
        
    for k, v in area_scores.items():
        snake_to_score[k] += v

    return snake_to_score, combined_board

  def get_area_control(self, game_state: dict) -> tuple:
    """
    Calculates area control for each snake based on the "shadow" concept and returns both
    scores and the control board.
    
    Args:
        game_state: full json dict of game data
    
    Returns:
        tuple: (dict of snake IDs to scores, numpy array of control board)
        Control board values:
         0: neutral space
         1: controlled by snake1
         2: controlled by snake2
    """
    """
    Calculates area control based on which snake is closest to each wall.
    """
    board_size = 11
    board = np.zeros((board_size, board_size), dtype=int)
    wall_control_board = np.zeros((board_size, board_size), dtype=int)
    snake_scores = defaultdict(float)
    
    # Get snakes
    snakes = game_state["Snakes"]
    if len(snakes) != 2:  # We need exactly 2 snakes for this calculation
        return 
    
    snake1, snake2 = snakes[0], snakes[1]
    
    # First mark snake bodies
    for i, snake in enumerate([snake1, snake2], 1):
        for segment in snake["Body"]:
            x, y = segment["X"], segment["Y"]
            if 0 <= x < board_size and 0 <= y < board_size:
                board[y, x] = i
                wall_control_board[y, x] = i
    
    # For each wall (top, right, bottom, left), find closest snake
    walls = {
        'top': {'y': board_size-1, 'direction': (-1, 0)},
        'right': {'x': board_size-1, 'direction': (0, -1)},
        'bottom': {'y': 0, 'direction': (1, 0)},
        'left': {'x': 0, 'direction': (0, 1)}
    }
    
    for wall_name, wall_info in walls.items():
        # Find minimum distance to wall for each snake
        snake1_dist = 0
        snake2_dist = 0
        
        # Check snake1's distance to wall
        for segment in snake1["Body"]:
            if wall_name in ['top', 'bottom']:
                dist = abs(wall_info['y'] - segment["Y"])
            else:
                dist = abs(wall_info['x'] - segment["X"])
            snake1_dist += dist
        
        # Check snake2's distance to wall
        for segment in snake2["Body"]:
            if wall_name in ['top', 'bottom']:
                dist = abs(wall_info['y'] - segment["Y"])
            else:
                dist = abs(wall_info['x'] - segment["X"])
            snake2_dist += dist
        
        # Determine which snake owns this wall
        owner = 1 if snake1_dist < snake2_dist else 2
        
        # Fill territory from wall inward
        dy, dx = wall_info['direction']
        if wall_name in ['top', 'bottom']:
            start_y = wall_info['y']
            for x in range(board_size):
                y = start_y
                hit_snake = False
                while 0 < y < board_size-1 and not hit_snake:
                    current_cell = board[y, x]
                    # Stop at any snake body
                    if current_cell != 0:
                        hit_snake = True
                        continue
                    if wall_control_board[y, x] == 0:  # Only fill empty spaces
                        wall_control_board[y, x] = owner - 5  # Using -4 and -3 for territory
                        snake_scores[snakes[owner-1]["ID"]] += 0.5
                    y += dy
        else:  # left or right wall
            start_x = wall_info['x']
            for y in range(board_size):
                x = start_x
                hit_snake = False
                while 0 < x < board_size-1 and not hit_snake:
                    current_cell = board[y, x]
                    # Stop at any snake body
                    if current_cell != 0:
                        hit_snake = True
                        continue
                    if wall_control_board[y, x] == 0:  # Only fill empty spaces
                        wall_control_board[y, x] = owner - 5  # Using -4 and -3 for territory
                        snake_scores[snakes[owner-1]["ID"]] += 0.5
                    x += dx
    
    board = np.zeros((board_size, board_size), dtype=int)
    directional_control_board = np.zeros((board_size, board_size), dtype=int)
    
    # Mark snake bodies on the board
    # 1 for snake1, 2 for snake2
    for i, snake in enumerate([snake1, snake2], 1):
        for segment in snake['Body']:
            x, y = segment['X'], segment['Y']
            
            # closest wall to segment
            dist_to_top = board_size - 1 - y
            dist_to_right = board_size - 1 - x
            dist_to_bottom = y
            dist_to_left = x
            
            # Find the nearest wall and its direction
            wall_distances = [
                (dist_to_top, (1, 0)),    # Up
                (dist_to_bottom, (-1, 0)),  # Right
                (dist_to_right, (0, 1)), # Down
                (dist_to_left, (0, -1))    # Left
            ]
            
            # Get the closest 2 directions
            wall_distances.sort(key=lambda x: x[0])
            directions = wall_distances[:2]
            # for each direction
            #    while there is no wall in that direction
            #        mark all spaces in that direction as snake territory
            for direction in directions:
              dy, dx = direction[1]
              fill_x, fill_y = x, y
              while fill_x > 0 and fill_x < board_size-1 and fill_y > 0 and fill_y < board_size-1:  # Empty space
                  directional_control_board[fill_y + dy, fill_x + dx] = i-5
                  fill_x += dx  # Continue in the same direction
                  fill_y += dy  # Continue in the same direction
                  snake_scores[snake['ID']] += 0.5
                  
            if 0 <= x < board_size and 0 <= y < board_size:
                board[y, x] = i
                directional_control_board[y, x] = i  # Snake bodies are automatically their territory
    
    board = np.zeros((board_size, board_size), dtype=int)
    area_control_board = np.zeros((board_size, board_size), dtype=int)
    
    # Calculate area control for each snake
    for y in range(board_size):
        for x in range(board_size):
            if board[y, x] == 0:  # Empty space
                # Calculate distances to nearest body segments of each snake
                min_dist_snake1 = float('inf')
                min_dist_snake2 = float('inf')
                
                # Check distance to snake1's body
                for segment in snake1['Body']:
                    dist = abs(x - segment['X']) + abs(y - segment['Y'])  # Manhattan distance
                    min_dist_snake1 = min(min_dist_snake1, dist)
                
                # Check distance to snake2's body
                for segment in snake2['Body']:
                    dist = abs(x - segment['X']) + abs(y - segment['Y'])
                    min_dist_snake2 = min(min_dist_snake2, dist)
                
                # Assign control based on relative distances
                if min_dist_snake1 < min_dist_snake2:
                    snake_scores[snake1['ID']] += 1
                    area_control_board[y, x] = 1
                elif min_dist_snake2 < min_dist_snake1:
                    snake_scores[snake2['ID']] += 1
                    area_control_board[y, x] = 2
                # If equal distances, remains neutral (0)
    
    # Normalize scores
    total_score = sum(snake_scores.values())
    if total_score > 0:
        for snake_id in snake_scores:
            snake_scores[snake_id] /= total_score
    
    return snake_scores, wall_control_board, directional_control_board, area_control_board

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

