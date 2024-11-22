from Data.data import Data 
from Model.model import Model 
from BattlesnakePipeline.battlesnake_pipeline import BattlesnakePipeline 

data_class = Data('./JSON')

battlesnake_model = Model(data_class)
# battlesnake_model.train_neural_network()
battlesnake_model.train_SVC()

battlesnake_pipeline = BattlesnakePipeline(data_class, battlesnake_model)

# raw_data = {'board':data_class.frames[5], 'you':{'ID':'gs_kyDB7QyKYDKVjKxQRjj7mG4M'}}

# print(battlesnake_pipeline.predict(raw_data))

battlesnake_pipeline.save("./model_pipeline")
