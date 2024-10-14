from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans  # or use GMM if preferred
def train_pca_kmeans(embeddings, n_clusters=500, n_components=100, save_path=None):
   
    pca = PCA(n_components=n_components)
    print("Fitting PCA")
    pca.fit(embeddings)
    pca_embeddings = pca.transform(embeddings)
    print("Fitting KMeans")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, max_iter=100, n_init=1, verbose=1, random_state=2, max_no_improvement=20, reassignment_ratio=0.01)
    kmeans.fit(pca_embeddings)
    print("PCA and KMeans trained, Kmeans inertia/datasize: ", kmeans.inertia_/len(embeddings))
    if save_path is not None:
        np.save(save_path.replace(".npy", "_kmeans.npy"), kmeans)
        np.save(save_path.replace(".npy", "_pca.npy"), pca)
    return pca, kmeans

def load_pca_kmeans(save_path):
    pca = np.load(save_path.replace(".npy", "_pca.npy"), allow_pickle=True).item()
    kmeans = np.load(save_path.replace(".npy", "_kmeans.npy"), allow_pickle=True).item()
    return pca, kmeans


def plot_cluster_embedding(success_embeddings, failure_embeddings)
    # Assuming you have success and failure embeddings
    # Combine both success and failure embeddings into one dataset
    embeddings = torch.concat((success_embeddings, failure_embeddings))

    # Perform t-SNE on the embeddings to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Perform KMeans clustering (or use GMM clustering instead)
    kmeans = KMeans(n_clusters=2, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Optionally, map predicted labels to colors for visualization
    # You can adjust this part if needed to match the true labels
    correct_labels = np.concatenate((np.ones(len(success_embeddings)), np.zeros(len(failure_embeddings))))

    # Plot the t-SNE transformed embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)

    # Add a legend if desired
    plt.colorbar(scatter, ticks=[0, 1])
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()