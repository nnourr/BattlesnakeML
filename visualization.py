from Data.data import Data
from Visualize.visualize import Visualize

# Initialize the Data class
data_class = Data(data_root='./JSON')

# Initialize the Visualize class with the Data instance
visualizer = Visualize(data_class)

# print(data_class.frames[0])
print(data_class.preprocessed[9])
# print(data_class.moves[0])
# print(data_class.encoded_moves[0])
# print(data_class.flattened_frames[0])

# print(data_class.preprocessed[50]['opp_distance']) # Distance of player head to opponent body

# print()
# print()
# print()

# print(data_class.preprocessed[50]['player_distance']) # Distance of opponent head to player body

# Use the Visualize class for plotting
visualizer.plot_lda('./vis_out/LDA/LDA_2D_relative.png', True)
# visualizer.plot_lda_3d('./vis_out/LDA/LDA_3D.png', False)  # No longer using 3D visualization
# visualizer.plot_lda_3d(None, True)

# visualizer.plot_mds(None, True)
# data_class.plot_mds('./vis_out/MDS_3d.png', False)
