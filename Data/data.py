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
        player_health = None
        enemy_health = False
        end_frame = False

        for snake in snakes:
            if (snake['Death'] != None):
               end_frame = True
            snake_body = [(s["X"], s["Y"]) for s in snake["Body"]]
            if snake["Author"] == author:
                player_body = snake_body
                player_health = snake["Health"]
                player_health = snake["Health"]
            else:
                enemy_body = snake_body
                enemy_health = snake["Health"]

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
            "valid_spaces": valid_spaces
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
    return np.append(self.flatten_frame_to_list(frame), [len(frame["player_body"]), len(frame["enemy_body"]), len(frame["food_positions"]), frame["health"], *self.flatten_frame_to_board(frame)], axis=0)
  
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

