import os
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Data:
  def __init__(self, data_root):
    self.data_root = data_root
    self.frames = self.__read_frames()
    self.preprocessed = self.__preprocess_frames()
    self.moves = self.__extract_moves()
    self.flattened_frames = self.__flatten_frames()
      
  def plot_mds(self, path = None, show = False):
    data_classes = set(['up', 'down', 'left', 'right'])
    
    euclidean_data = self.__calculate_euclidean_distance()
    mds = self.__multi_dim_scaling(euclidean_data)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    plt.title('MDS Representation of Battlesnake Moves')

    unique_classes = list(set(label['Move'] for label in self.moves))
    colors = plt.cm.get_cmap('tab10', len(unique_classes))  # Using a colormap with enough distinct colors
    class_colors = {cls: colors(i) for i, cls in enumerate(unique_classes)}

    for i, label in enumerate(self.moves):
        cls = label['Move']
        
        # Plot each point with its respective color
        ax.scatter(mds[i, 0], mds[i, 1], color=class_colors[cls], s=10)
        
        # Annotate each point with its index
        ax.annotate(str(i),
                    (mds[i, 0], mds[i, 1]),            # Point to annotate
                    textcoords="offset points",        # Position relative to the point
                    xytext=(5, 5),                     # Offset of the annotation text
                    ha='center', color='red') 

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

        for snake in snakes:

            snake_body = [(s["X"], s["Y"]) for s in snake["Body"]]
            if snake["Author"] == author:
                player_body = snake_body
            else:
                enemy_body = snake_body

        if player_body is None:
            continue  # Skip if the player's snake is not found

        # Flatten features into a vector (simple representation)
        features = {
            "turn": turn,
            "food_positions": [(f["X"], f["Y"]) for f in food],
            "player_body": player_body,
            "enemy_body": enemy_body
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
  
  def __flatten_frames(self):
    flattened_frames = []
    for i, frame in enumerate(self.preprocessed):
      flattened_frames.append(self.__flatten_frame(frame, i+1))
      
    return np.array(flattened_frames)
  
  def __flatten_frame(self, frame, i):
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
  
  def __calculate_euclidean_distance(self):
    pairwise_distances = pdist(self.flattened_frames, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, 0)  # Ensuring D(a, a) = 0
    return distance_matrix
  
  def __multi_dim_scaling(self, data, dim = 2):
    mds = MDS(n_components=dim, random_state=0, dissimilarity='precomputed')  
    return mds.fit_transform(data)
