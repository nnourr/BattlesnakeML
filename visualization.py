from Data.data import Data 

data_class = Data('./JSON')
# print(data_class.preprocessed[0])
print(data_class.preprocessed[1])
print(data_class.moves[0])
print(data_class.flattened_frames[34])

data_class.plot_mds('./vis_out/MDS_1.png', False)
