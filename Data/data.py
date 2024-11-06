import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
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
matplotlib.use('TkAgg')


class Data:
  def __init__(self, data_root):
    self.data_root = data_root
    self.frames = self.__read_frames()
    self.preprocessed = self.__preprocess_frames()
    self.moves = self.__extract_moves()
    self.encoded_moves = self.__encode_moves()
    self.flattened_frames = self.__flatten_frames()
    
  def plot_lda(self, path = None, show = False):
    X = self.flattened_frames[:-1] # last frame has no move
    y = self.encoded_moves

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test for more realistic scenario (optional)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit_transform(X_scaled, y)
    
    # Plot the results
    plt.figure(figsize=(15, 15))

    target_names = ['up', 'down', 'left', 'right']
    labels = [0, 1, 2, 3]
    colors = ['blue', 'green', 'red', 'purple']
    for color, label in zip(colors, labels):
        idx = y == label
        plt.scatter(X_r2[idx, 0], X_r2[idx, 1], color=color, label=target_names[label], alpha=0.7)

    plt.title("LDA Representation of Battlesnake Moves")
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.legend(loc="best")
    
    if (show):
      plt.show()
     
    if(path !=  None): 
      plt.savefig(path, format='png', bbox_inches='tight', dpi=400) # PNG
    plt.clf()
    
    score = silhouette_score(X_r2, y)
    print(f'Silhouette Score after LDA: {score}')
    
    db_index = davies_bouldin_score(X_r2, y)
    print(f'Davies-Bouldin Index after LDA: {db_index}')
      
  def plot_lda_3d(self, path=None, show=False):
    X = self.flattened_frames[:-1]  # Last frame has no move
    y = self.encoded_moves

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize LDA with 3 components for 3D plotting
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_r3 = lda.fit_transform(X_scaled, y)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    target_names = ['up', 'down', 'left', 'right']
    labels = [0, 1, 2, 3]
    colors = ['blue', 'green', 'red', 'purple']
    for color, label in zip(colors, labels):
        idx = y == label
        ax.scatter(X_r3[idx, 0], X_r3[idx, 1], X_r3[idx, 2], color=color, label=target_names[label], alpha=0.7)

    # Set labels and title
    ax.set_title("3D LDA Representation of Battlesnake Moves")
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    ax.set_zlabel("LDA Component 3")
    ax.legend(loc="best")
    
    # Show or save the plot
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
    plt.clf()
    
    # Calculate and print clustering scores
    score = silhouette_score(X_r3, y)
    print(f'Silhouette Score after 3D LDA: {score}')
    
    db_index = davies_bouldin_score(X_r3, y)
    print(f'Davies-Bouldin Index after 3D LDA: {db_index}')

  
  def plot_mds(self, path = None, show = False):    
    euclidean_data = self.__calculate_euclidean_distance()
    mds = self.__multi_dim_scaling(euclidean_data)
    
    fig = plt.figure(figsize=(15,15))
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    
    plt.title('MDS Representation of Battlesnake Moves')

    unique_classes = list(set(label['Move'] for label in self.moves))
    colors = plt.cm.get_cmap('tab10', len(unique_classes))  # Using a colormap with enough distinct colors
    class_colors = {cls: colors(i) for i, cls in enumerate(unique_classes)}

    for i, label in enumerate(self.moves):
        cls = label['Move']
        
        # Plot each point with its respective color
        ax.scatter(mds[i, 0], mds[i, 1], color=class_colors[cls], s=10)
        # ax.scatter(mds[i, 0], mds[i, 1], mds[i,2], color=class_colors[cls], s=10)
        
        # Annotate each point with its index
        ax.annotate(str(i),
                    (mds[i, 0], mds[i, 1]),            # Point to annotate
                    textcoords="offset points",        # Position relative to the point
                    xytext=(5, 5),                     # Offset of the annotation text
                    ha='center', color='red') 
        # ax.text(mds[i, 0], mds[i, 1], mds[i, 2], str(i),
        #     color='red', ha='center')

    for cls, color in class_colors.items():
      ax.scatter([], [], color=color, label=cls)

    # Add legend and show the plot
    ax.legend(title="Classes")

    if (show):
      plt.show()
     
    if(path !=  None): 
      plt.savefig(path, format='png', bbox_inches='tight', dpi=400) # PNG
    plt.clf()
  
  def __read_frames(self):
    frames = []
    for item in os.listdir(self.data_root):
        item_path = os.path.join(self.data_root, item)
        frames += self.__read_frames_file(item_path)
    return frames
  
  def __read_frames_file(self, file_path):
    with open(file_path, 'r') as _file:
      frames = json.load(_file)
    
    return frames
  
  def __preprocess_frames(self, author='coreyja'):
    preprocessed = []

    for game_state in self.frames:
        turn = game_state["Turn"]
        food = game_state["Food"]
        snakes = game_state["Snakes"]

        # Initialize lists for player and enemy snakes
        player_body = None
        enemy_body = None
        player_health = None

        for snake in snakes:

            snake_body = [(s["X"], s["Y"]) for s in snake["Body"]]
            if snake["Author"] == author:
                player_body = snake_body
                player_health = snake["Health"]
            else:
                enemy_body = snake_body

        if player_body is None:
            continue  # Skip if the player's snake is not found

        # Flatten features into a vector (simple representation)
        features = {
            "turn": turn,
            "food_positions": [(f["X"], f["Y"]) for f in food],
            "player_body": player_body,
            "enemy_body": enemy_body,
            "health": player_health,
        }
        preprocessed.append(features)

    return preprocessed
  
  def __extract_moves(self):
    moves = []

    # Initialize previous head positions for the given snakeID
    previous_position = ''

    for i, turn_data in enumerate(self.preprocessed):
        player_snake = turn_data['player_body']

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

    return moves
  
  def __encode_moves(self):
    action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    return np.array([action_map[move['Move']] for move in self.moves])
  
  def __flatten_frames(self):
    flattened_frames = []
    for i, frame in enumerate(self.preprocessed):
      flattened_frames.append(self.__flatten_frame_i_to_list(frame, i+1))
      
    return np.array(flattened_frames)
  
  def __flatten_frame_i_to_list(self, frame, i):

    # add length to arrays
    return np.append(self.__flatten_frame_to_list(frame, i), [len(frame["player_body"]), len(frame["enemy_body"]), len(frame["food_positions"]), i, frame["health"]], axis=0)
    # return np.concatenate([food_positions_padded, player_body_padded, enemy_bodies_padded])
  
  def __flatten_frame_to_list(self, frame, i):
    food_positions = [coord for f in frame["food_positions"] for coord in f]
    player_body = [coord for s in frame["player_body"] for coord in s]
    enemy_bodies = [coord for b in frame["enemy_body"] for coord in b]
     # Define the desired length for padding
    desired_length = 100

    # Function to pad arrays to the desired length
    def pad_to_length(arr, length):
        return np.pad(arr, (0, max(0, length - len(arr))), mode='constant')

    # Pad each feature array to the desired length
    # turn_padded = pad_to_length(turn, desired_length)
    food_positions_padded = pad_to_length(food_positions, desired_length)
    player_body_padded = pad_to_length(player_body, desired_length)
    enemy_bodies_padded = pad_to_length(enemy_bodies, desired_length)

    # Concatenate all arrays
    return np.concatenate([food_positions_padded, player_body_padded, enemy_bodies_padded])
  
  def __flatten_frame_to_board(self, frame, i):
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
  
  def __calculate_euclidean_distance(self):
    pairwise_distances = pdist(self.flattened_frames, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, 0)  # Ensuring D(a, a) = 0
    return distance_matrix
  
  def __multi_dim_scaling(self, data, dim = 2):
    mds = MDS(n_components=dim, random_state=0, dissimilarity='precomputed')  
    return mds.fit_transform(data)

