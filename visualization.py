from Data.data import Data 

data_class = Data('./JSON')
# print(data_class.preprocessed[0])
print(data_class.preprocessed[1])
print(data_class.moves[0])
print(data_class.flattened_frames[34])

# data_class.plot_lda('./vis_out/LDA/LDA_scaled_5_health.png', False)
# data_class.plot_lda(None, True)

# data_class.plot_lda_3d('./vis_out/LDA/LDA_scaled_3d_3_health.png', False)
data_class.plot_lda_3d(None, True)

# data_class.plot_mds(None, True)
# data_class.plot_mds('./vis_out/MDS_3d.png', False)
