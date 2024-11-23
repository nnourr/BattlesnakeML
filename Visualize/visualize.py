import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

class Visualize:
    def __init__(self, data_instance):
        self.data_instance = data_instance

    def plot_lda(self, path=None, show=False):
        X = self.data_instance.flattened_frames
        y = self.data_instance.encoded_moves

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit_transform(X_scaled, y)
        plt.figure(figsize=(15, 15))
        target_names = ['ahead', 'turn_left', 'turn_right']
        labels = [0, 1, 2]
        colors = ['blue', 'green', 'red']
        for color, label in zip(colors, labels):
            idx = y == label
            plt.scatter(X_r2[idx, 0], X_r2[idx, 1], color=color, label=target_names[label], alpha=0.7)
        plt.title("LDA Representation of Relative Battlesnake Moves")
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
        X = self.data_instance.flattened_frames
        y = self.data_instance.encoded_moves

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        lda = LinearDiscriminantAnalysis(n_components=2)  
        X_r2 = lda.fit_transform(X_scaled, y)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)  
        target_names = ['ahead', 'turn_left', 'turn_right']
        labels = [0, 1, 2]
        colors = ['blue', 'green', 'red']
        for color, label in zip(colors, labels):
            idx = y == label
            ax.scatter(X_r2[idx, 0], X_r2[idx, 1], color=color, label=target_names[label], alpha=0.7)
        ax.set_title("2D LDA Representation of Relative Battlesnake Moves")
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("LDA Component 2")
        ax.legend(loc="best")
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=400)
        plt.clf()
        score = silhouette_score(X_r2, y)
        print(f'Silhouette Score after 2D LDA: {score}')
        db_index = davies_bouldin_score(X_r2, y)
        print(f'Davies-Bouldin Index after 2D LDA: {db_index}')

    def plot_mds(self, path=None, show=False):
        euclidean_data = self.data_instance.calculate_euclidean_distance()
        mds = self.data_instance.multi_dim_scaling(euclidean_data, n_components=3)  
        moves = self.data_instance.moves

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')  
        plt.title('3D MDS Representation of Relative Battlesnake Moves')
        
        move_to_class = {
            'up': 'ahead',
            'down': 'ahead',
            'left': 'ahead',
            'right': 'ahead'
        }
        
        for i in range(len(moves)):
            if i > 0:  
                current_move = moves[i]['Move']
                prev_move = moves[i-1]['Move']
                
                if current_move == prev_move:
                    move_to_class[current_move] = 'ahead'
                elif (prev_move == 'up' and current_move == 'left') or \
                     (prev_move == 'left' and current_move == 'down') or \
                     (prev_move == 'down' and current_move == 'right') or \
                     (prev_move == 'right' and current_move == 'up'):
                    move_to_class[current_move] = 'turn_left'
                else:
                    move_to_class[current_move] = 'turn_right'

        class_colors = {
            'ahead': 'blue',
            'turn_left': 'green',
            'turn_right': 'red'
        }

        for i, label in enumerate(moves):
            move = label['Move']
            relative_move = move_to_class[move]
            color = class_colors[relative_move]
            ax.scatter(mds[i, 0], mds[i, 1], mds[i, 2], color=color, s=50)

        for move_class, color in class_colors.items():
            ax.scatter([], [], [], color=color, label=move_class)
        
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        ax.set_zlabel('MDS Component 3')
        ax.legend(title="Relative Moves")
        
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
        X = self.data_instance.flattened_frames[:-1]  
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