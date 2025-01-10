import umap
import matplotlib.pyplot as plt
import torch
import numpy as np

from data_preprocessing.mumin import load_mumin_heterodata
from network_analysis.important_nodes_analysis_v2 import user_claim_discussion_stats


def plot_user_embeddings_with_claims(user_embeddings, user_stats):
    # Ensure user_embeddings is a numpy array
    if isinstance(user_embeddings, torch.Tensor):
        user_embeddings = user_embeddings.cpu().numpy()  # Move to CPU before converting to NumPy

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(user_embeddings)

    # Compute colors for each user based on the true/false claims discussed
    colors = []
    for user_id in range(len(user_embeddings)):  # Assuming user IDs align with the embedding index
        # Find the user stats for this user_id
        true_claims, false_claims = next(((item[1], item[2]) for item in user_stats if len(item)>=3 and item[0] == user_id), (0, 0))

        total_claims = true_claims + false_claims

        if total_claims == 0:
            # User has not discussed any claims; use white
            colors.append((1.0, 1.0, 1.0))  # RGB for white
        else:
            true_ratio = true_claims / total_claims
            false_ratio = false_claims / total_claims
            # Gradation: Blue for true, Red for false, White for mixed
            blue_intensity = true_ratio
            red_intensity = false_ratio
            colors.append((red_intensity, 1 - (blue_intensity + red_intensity), blue_intensity))

    # Convert colors to numpy array for plotting
    colors = np.array(colors)
    print(f"Number of colors: {len(colors)}")

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=10, alpha=0.8)
    plt.title("User Embeddings with Claim Discussion Colors")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Add a custom color legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                   label='Mostly True (Dark Blue)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                   label='Mostly False (Dark Red)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, label='Mixed (White)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.show()



heterodata = load_mumin_heterodata()
user_embeddings = heterodata['user'].x
num_users = user_embeddings.shape[0]

user_stats = user_claim_discussion_stats(heterodata, num_users)
plot_user_embeddings_with_claims(user_embeddings, user_stats)
