#Alternative to training: load saved model
def load_model(hidden_channels,out_channels,dropout,data,weigths_filename):
    model = GAT(hidden_channels, out_channels,dropout)
    model = to_hetero(model, data.metadata(), aggr='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)
    with torch.no_grad():
        model.eval()
        model(data.x_dict, data.edge_index_dict)
        model.train()
    model.load_state_dict(torch.load(weigths_filename))
    return model
