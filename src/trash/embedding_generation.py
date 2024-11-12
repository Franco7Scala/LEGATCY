import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx


# Step 1: Create and preprocess the NetworkX directed graph
def create_directed_graph():
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node(0, features=[1.0, 2.0, 3.0])  # Example features
    G.add_node(1, features=[2.0, 3.0, 1.0])
    G.add_node(2, features=[3.0, 1.0, 2.0])

    # Add weighted edges
    G.add_edge(0, 1, weight=0.5)
    G.add_edge(1, 2, weight=0.8)
    G.add_edge(2, 0, weight=1.0)

    return G


# Convert NetworkX graph to PyTorch Geometric format
def convert_to_torch_geometric(G):
    # Ensure the graph has a 'features' attribute on each node
    for node in G.nodes(data=True):
        if 'features' not in node[1]:
            raise ValueError("Each node must have a 'features' attribute with a list of feature values.")

    # Convert node features to PyTorch tensors
    for node in G.nodes:
        G.nodes[node]['x'] = torch.tensor(G.nodes[node]['features'], dtype=torch.float)

    # Convert edge weights to a 'weight' attribute
    for u, v, data in G.edges(data=True):
        data['weight'] = torch.tensor(data.get('weight', 1.0), dtype=torch.float)

    # Use from_networkx to convert NetworkX graph to PyTorch Geometric Data format
    data = from_networkx(G)

    # Ensure correct edge attributes (weight)
    if 'weight' in data.edge_attr:
        data.edge_weight = data.edge_attr.squeeze()  # Remove any extra dimensions

    return data


# Step 2: Define the GATv2 model architecture
class GATv2EmbeddingModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATv2EmbeddingModel, self).__init__()
        self.gatv2_conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gatv2_conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.gatv2_conv1(x, edge_index, edge_weight=edge_weight))
        x = self.gatv2_conv2(x, edge_index, edge_weight=edge_weight)
        return x


# Step 3: Train the GATv2 model to generate node embeddings
def train_gatv2(data, in_channels, hidden_channels, out_channels, epochs=200, learning_rate=0.005):
    model = GATv2EmbeddingModel(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # For unsupervised node embeddings, we minimize the sum of edge weights for adjacent nodes.
        # This is a simple unsupervised objective, but you may replace this with other tasks.
        loss = F.mse_loss(out[data.edge_index[0]], out[data.edge_index[1]], reduction='mean')
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


# Step 4: Extract embeddings for all nodes
def get_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
    return embeddings


# Example usage
if __name__ == "__main__":
    # Step 1: Create and preprocess graph
    G = create_directed_graph()

    # Step 2: Convert to PyTorch Geometric data format
    data = convert_to_torch_geometric(G)

    # Specify feature dimensions
    in_channels = data.x.size(1)  # Dimension of node features
    hidden_channels = 32  # Dimension of hidden layer (configurable)
    out_channels = 64  # Dimension of output embeddings (configurable)

    # Step 3: Train the GATv2 model
    model = train_gatv2(data, in_channels, hidden_channels, out_channels)

    # Step 4: Extract and print node embeddings
    embeddings = get_node_embeddings(model, data)
    print("Node embeddings:\n", embeddings)
