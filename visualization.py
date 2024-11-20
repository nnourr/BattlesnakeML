from Data.data import Data
from Visualize.visualize import Visualize

# Initialize the Data class
data_class = Data(data_root='./JSON')

# Initialize the Visualize class with the Data instance
visualizer = Visualize(data_class)

# print(data_class.frames[0])
# print(data_class.preprocessed[0])
# print(data_class.moves[0])
# print(data_class.encoded_moves[0])
print(data_class.flattened_frames[0])

# Use the Visualize class for plotting
# visualizer.plot_lda('./vis_out/LDA/LDA_2D.png', False)
# visualizer.plot_lda_3d('./vis_out/LDA/LDA_3D.png', False)
# visualizer.plot_mds('./vis_out/MDS/MDS_3D.png', False)

visualizer.plot_tsne(path='./vis_out/TSNE/TSNE_2D.png', show=False)

# Uncomment to show plots
# visualizer.plot_lda(None, True)