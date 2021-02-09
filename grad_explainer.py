import numpy as np
import torch
from torch_geometric.nn import MessagePassing


def execute_model_with_gradient(model, node, x, edge_index):
    ypred = model(x, edge_index)

    predicted_labels = ypred.argmax(dim=-1)
    predicted_label = predicted_labels[node]
    logit = torch.nn.functional.softmax((ypred[node, :]).squeeze(), dim=0)

    logit = logit[predicted_label]
    loss = -torch.log(logit)
    loss.backward()


def grad_edge_explanation(model, node, x, edge_index):
    model.zero_grad()

    E = edge_index.size(1)
    edge_mask = torch.nn.Parameter(torch.ones(E))

    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edge_mask

    edge_mask.requires_grad = True
    x.requires_grad = True

    if edge_mask.grad is not None:
        edge_mask.grad.zero_()
    if x.grad is not None:
        x.grad.zero_()

    execute_model_with_gradient(model, node, x, edge_index)

    adj_grad = edge_mask.grad
    adj_grad = torch.abs(adj_grad)
    masked_adj = adj_grad + adj_grad.t()
    masked_adj = torch.sigmoid(masked_adj)
    masked_adj = masked_adj.cpu().detach().numpy()

    feature_mask = torch.abs(x.grad).cpu().detach().numpy()

    return np.max(feature_mask, axis=0), masked_adj


def grad_node_explanation(model, node, x, edge_index):
    model.zero_grad()

    num_nodes, num_features = x.size()

    node_grad = torch.nn.Parameter(torch.ones(num_nodes))
    feature_grad = torch.nn.Parameter(torch.ones(num_features))

    node_grad.requires_grad = True
    feature_grad.requires_grad = True

    mask = node_grad.unsqueeze(0).T.matmul(feature_grad.unsqueeze(0))

    execute_model_with_gradient(model, node, mask*x, edge_index)

    node_mask = torch.abs(node_grad.grad).cpu().detach().numpy()
    feature_mask = torch.abs(feature_grad.grad).cpu().detach().numpy()

    return feature_mask, node_mask


def gradinput_node_explanation(model, node, x, edge_index):
    model.zero_grad()

    x.requires_grad = True
    if x.grad is not None:
        x.grad.zero_()

    execute_model_with_gradient(model, node, x, edge_index)

    feature_mask = torch.abs(x.grad * x).cpu().detach().numpy()

    return np.mean(feature_mask, axis=0), np.mean(feature_mask, axis=1)
