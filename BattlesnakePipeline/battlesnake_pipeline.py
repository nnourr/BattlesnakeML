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
    
    return self.data.decode_moves(predictions)[0]
  
  def __preprocess(self, raw_data):
    game_state = raw_data['board']
    turn = game_state["Turn"]
    food = game_state["Food"]
    food = [(f["X"], f["Y"]) for f in food]
    snakes = game_state["Snakes"]

    # Initialize lists for player and enemy snakes
    player_body = None
    enemy_body = None
    player_health = None
    
    player_snake_id = raw_data['you']['ID']

    for snake in snakes:
        snake_body = [(s["X"], s["Y"]) for s in snake["Body"]]
        if snake["ID"] == player_snake_id: 
          player_body = snake_body
          player_health = snake["Health"]
          continue
        
        enemy_body = snake_body

    # Calculate the distance between the player's snake head and the enemy's snake head
    player_head = player_body[0]
    enemy_head = enemy_body[0]

    opp_distance = self.data.calculate_snake_distance(player_head, enemy_body)
    player_distance = self.data.calculate_snake_distance(enemy_head, player_body)
    pairwise_distance = self.data.calculate_pairwise_distance(enemy_body, player_body)
    wall_distance = self.data.calculate_wall_distance(player_body)
    food_distance = self.data.calculate_food_distance(player_body, food)
    valid_spaces = self.data.calculate_valid_moves(player_body, enemy_body)

    # Flatten features into a vector (simple representation)
    features = {
        "turn": turn,
        "food_positions": food,
        "player_body": player_body,
        "enemy_body": enemy_body,
        "health": player_health,
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