import torch.nn as nn
import torch

class Local_GNN(nn.Module):
    def __init__(self, A, input_window, hidden_dim):
        super(Local_GNN, self).__init__()
        self.input_window = input_window
        self.adj = A
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        g_out = self.act(self.lin(torch.einsum('vn,btnd->btvd', self.adj, x)))
        return g_out

class DMVSTNet(nn.Module):
    def __init__(self, args, device, dim_in, dim_out):
        super().__init__()
        self.adj_mx = args.adj_mx.to(device)
        self.device = device
        self.num_nodes = args.num_nodes
        self.hidden_dim = args.hidden_dim
        self.dim_out = dim_out
        self.topo_embedded_dim = args.topo_embedded_dim

        self.input_window = args.input_window
        self.output_window = args.output_window

        self.Lin_in_spa = nn.Linear(dim_in, self.hidden_dim)
        self.Lin_in_tem = nn.Linear(dim_in, self.hidden_dim)
        self.Lin_in_sen = nn.Linear(dim_in, self.hidden_dim)
        self.Local_GNN1 = Local_GNN(self.adj_mx, self.input_window, self.hidden_dim)
        self.Local_GNN2 = Local_GNN(self.adj_mx, self.input_window, self.hidden_dim)
        self.Local_GNN3 = Local_GNN(self.adj_mx, self.input_window, self.hidden_dim)
        self.Lin_spa = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.hidden_dim*dim_out, hidden_size=self.hidden_dim*dim_out, batch_first=True, num_layers=1)
        self.node_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.topo_embedded_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.w = nn.Parameter(torch.empty(self.topo_embedded_dim, self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.w)
        self.output = nn.Linear(in_features=self.hidden_dim*dim_out+self.hidden_dim, out_features=dim_out)

    def forward(self, x):
        x_in_spa = self.Lin_in_spa(x)
        x_in_tem = self.Lin_in_tem(x)
        x_in_sen = self.Lin_in_sen(x)
        Local_gnn_out1 = self.Local_GNN1(x_in_spa)
        spatial_out = self.Lin_spa(Local_gnn_out1) + x_in_spa
        ret = torch.cat([spatial_out, x_in_tem], dim=-1)
        ret = ret.permute(0, 2, 1, 3).contiguous()

        ret = ret.view(-1, self.input_window, self.hidden_dim*self.dim_out)
        out_lstm, (hid, cell) = self.lstm(ret)
        hid_out = hid.mean(0).squeeze(0).view(-1, self.num_nodes, 1, self.hidden_dim*self.dim_out)
        temporal_out = ((out_lstm.squeeze(1)).view(spatial_out.shape[0], self.num_nodes, -1, self.hidden_dim*self.dim_out) + hid_out).transpose(1, 2)

        weights = torch.einsum('nd,dio->nio', self.node_embeddings, self.w)
        x_out_sen = torch.einsum('btni,nio->btno', x_in_sen, weights)

        static_dynamic_concate = torch.cat([temporal_out, x_out_sen], dim=-1)
        out = self.output(static_dynamic_concate)

        return out
