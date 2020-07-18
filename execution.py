import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from pathlib import Path

from explainer import *

dataset = Planetoid(root='./tmp/Cora', name='Cora')
Path("./results/cora").mkdir(parents=True, exist_ok=True)
path_to_saved_model = "./results/cora/gcn_2_layers.pt"
path_to_saved_explanation_prefix = "./results/cora/gcn_2_layers_explanation_"


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def load_model(path, model):
    model.load_state_dict(torch.load(path))
    model.eval()


def train_model(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def print_accuracy(model, data):
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model.to(device)
data = dataset[0].to(device)

try:
    load_model(path_to_saved_model, model)
except FileNotFoundError:
    train_model(model, data)
    save_model(model, path_to_saved_model)

print_accuracy(model, data)

explainer = SISDistortionGraphExplainer(model)
number_of_nodes, _ = data.x.size()

for node in range(number_of_nodes):
    explanation = explainer.explain_node(node, data.x, data.edge_index)

    save_minimal_nodes_and_features_sets(path_to_saved_explanation_prefix, node, explanation)
