from Data.data import Data 
from Model.model import Model 

data_class = Data('./JSON')

battlesnake = Model(data_class)
# battlesnake.train_neural_network()
battlesnake.train_SVC()
