import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BattlesnakePipeline(BaseEstimator, TransformerMixin):
  def __init__(self, data, model):
    self.lda = data.lda_transform
    self.data = data
    self.model = model.model
    self.preprocess = data.flatten_frame_i_to_list
    pass
  
  def predict(self, raw_data):
    """
    Predict the output for new raw data.
    
    Args:
        raw_data: Raw input data (before preprocessing).
    
    Returns:
        Predictions from the SVM model.
    """
    # Step 1: Preprocess raw data
    preprocessed_data = self.__preprocess(raw_data)
    preprocessed_data = np.array(preprocessed_data, dtype=float)
    for i, value in enumerate(preprocessed_data.flatten()):
      if value != value:  # NaN values are not equal to themselves
          print(f"NaN detected at index {i}")
    if preprocessed_data.ndim == 1:
      preprocessed_data = preprocessed_data.reshape(1, -1)  # Reshape to 2D
    assert not np.isnan(preprocessed_data).any(), "Data contains NaN values."
    assert not np.isinf(preprocessed_data).any(), "Data contains infinite values."
    
    # Step 2: Transform using the LDA model
    transformed_data = self.lda(preprocessed_data)
    
    # Step 3: Predict using the SVM model
    predictions = self.model.predict(transformed_data)
    
    encoded_move = predictions[0]
    valid_moves = self.data.calculate_valid_moves(self.player_body, self.enemy_body)
    
    print(valid_moves)
    print(encoded_move)
    
    if (valid_moves[encoded_move] == 0):
      print("UHH OHHH, INVALID MOVE") 
      encoded_move = valid_moves.index(1)
    
    move = self.data.decode_moves([encoded_move])
    print(move)
    return move[0]
    
  
  def __preprocess(self, raw_data):
    is_upper = 'Board' in raw_data
    def to_case(key):
      return key.title() if is_upper else key
    
    game_state = raw_data[to_case("board")]
    turn = raw_data[to_case("turn")]
    food = game_state[to_case("food")]
    food = [(f[to_case("x")], f[to_case("y")]) for f in food]
    snakes = game_state[to_case("snakes")]

    # Initialize lists for player and enemy snakes
    player_body = None
    enemy_body = None
    player_health = None
    enemy_health = None
    
    player_snake_id = raw_data[to_case("you")][to_case("id")]

    for snake in snakes:
        snake_body = [(s[to_case("x")], s[to_case("y")]) for s in snake[to_case("body")]]
        if snake[to_case("id")] == player_snake_id: 
          player_body = snake_body
          player_health = snake[to_case("health")]
          continue
        enemy_health = snake[to_case("health")]
        enemy_body = snake_body
        
    if enemy_body == None or player_body == None:
      return [0]

    # Calculate the distance between the player's snake head and the enemy's snake head
    player_head = player_body[0]
    enemy_head = enemy_body[0]

    opp_distance = self.data.calculate_snake_distance(player_head, enemy_body)
    player_distance = self.data.calculate_snake_distance(enemy_head, player_body)
    pairwise_distance = self.data.calculate_pairwise_distance(enemy_body, player_body)
    wall_distance = self.data.calculate_wall_distance(player_body)
    food_distance = self.data.calculate_food_distance(player_body, food)
    valid_spaces = self.data.calculate_valid_moves(player_body, enemy_body)

    self.player_body = player_body
    self.enemy_body = enemy_body
    
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
    
    return self.preprocess(features)
  
  def save(self, filepath):
    """
    Save the pipeline to a file.
    
    Args:
        filepath: File path to save the pipeline.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(self, f)

  @staticmethod
  def load(filepath):
      """
      Load the pipeline from a file.
      
      Args:
          filepath: File path to load the pipeline from.
      
      Returns:
          Loaded pipeline object.
      """
      with open(filepath, 'rb') as f:
          return pickle.load(f)