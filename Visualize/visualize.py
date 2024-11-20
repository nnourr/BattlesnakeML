import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

class Visualize:
    def __init__(self, data_instance):
        self.data_instance = data_instance

    def plot_lda(self, path=None, show=False):
        X = self.data_instance.flattened_frames[:-1]  # Last frame has no move
        y = self.data_instance.encoded_moves

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit_transform(X_scaled, y)
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
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
        plt.clf()
        score = silhouette_score(X_r2, y)
        print(f'Silhouette Score after LDA: {score}')
        db_index = davies_bouldin_score(X_r2, y)
        print(f'Davies-Bouldin Index after LDA: {db_index}')

    def plot_lda_3d(self, path=None, show=False):
        X = self.data_instance.flattened_frames[:-1]  # Last frame has no move
        y = self.data_instance.encoded_moves

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        lda = LinearDiscriminantAnalysis(n_components=3)
        X_r3 = lda.fit_transform(X_scaled, y)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        target_names = ['up', 'down', 'left', 'right']
        labels = [0, 1, 2, 3]
        colors = ['blue', 'green', 'red', 'purple']
        for color, label in zip(colors, labels):
            idx = y == label
            ax.scatter(X_r3[idx, 0], X_r3[idx, 1], X_r3[idx, 2], color=color, label=target_names[label], alpha=0.7)
        ax.set_title("3D LDA Representation of Battlesnake Moves")
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("LDA Component 2")
        ax.set_zlabel("LDA Component 3")
        ax.legend(loc="best")
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
        plt.clf()
        score = silhouette_score(X_r3, y)
        print(f'Silhouette Score after 3D LDA: {score}')
        db_index = davies_bouldin_score(X_r3, y)
        print(f'Davies-Bouldin Index after 3D LDA: {db_index}')

    def plot_mds(self, path=None, show=False):
        euclidean_data = self.data_instance.calculate_euclidean_distance()
        mds = self.data_instance.multi_dim_scaling(euclidean_data)
        moves = self.data_instance.moves

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot()
        plt.title('MDS Representation of Battlesnake Moves')
        unique_classes = list(set(label['Move'] for label in moves))
        colors = plt.cm.get_cmap('tab10', len(unique_classes))
        class_colors = {cls: colors(i) for i, cls in enumerate(unique_classes)}
        for i, label in enumerate(moves):
            cls = label['Move']
            ax.scatter(mds[i, 0], mds[i, 1], color=class_colors[cls], s=10)
            ax.annotate(str(i), (mds[i, 0], mds[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center', color='red')
        for cls, color in class_colors.items():
            ax.scatter([], [], color=color, label=cls)
        ax.legend(title="Classes")
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
        plt.clf()

    def plot_tsne(self, perplexity=30, max_iter=1000, path=None, show=False):
        """
        Perform t-SNE visualization on the data from the Data instance.

        Parameters:
        - perplexity: float, optional (default: 30)
        - max_iter: int, optional (default: 1000)
        - path: str, optional (default: None)
        - show: bool, optional (default: False)

        Returns:
        - None
        """
        X = self.data_instance.flattened_frames[:-1]  # Last frame has no move
        y = self.data_instance.encoded_moves

        print(X[0])
        print(y)

        tsne = TSNE(perplexity=perplexity, max_iter=max_iter)
        transformed_data = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        target_names = ['up', 'down', 'left', 'right']
        labels = [0, 1, 2, 3]
        colors = ['blue', 'green', 'red', 'purple']
        for color, label in zip(colors, labels):
            idx = y == label
            plt.scatter(transformed_data[idx, 0], transformed_data[idx, 1], color=color, label=target_names[label], alpha=0.7)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(loc="best")
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
        plt.clf()