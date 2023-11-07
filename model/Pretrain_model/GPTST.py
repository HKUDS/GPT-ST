import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MLP_RL(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, embed_dim, device):
        super(MLP_RL, self).__init__()

        self.ln1 = nn.Linear(dim_in, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, dim_out)

        self.weights_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))

        self.weights_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))
        self.act = nn.LeakyReLU()
        self.device = device

    def forward(self, eb, time_eb, node_eb):
        eb_out = self.ln1(eb)

        weights_spa = torch.einsum('nd,dio->nio', node_eb, self.weights_pool_spa)
        bias_spa = torch.matmul(node_eb, self.bias_pool_spa)
        out_spa = torch.einsum('btni,nio->btno', eb_out, weights_spa) + bias_spa
        out_spa = self.act(out_spa)

        weights_tem = torch.einsum('btd,dio->btio', time_eb, self.weights_pool_tem)
        bias_tem = torch.matmul(time_eb, self.bias_pool_tem).unsqueeze(-2)
        out_tem = torch.einsum('btni,btio->btno', out_spa, weights_tem) + bias_tem
        out_tem = self.act(out_tem)
        logits = self.ln3(out_tem)
        return logits

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

class cap_adj(nn.Module):
    def __init__(self, dim, num_nodes, timesteps, embed_dim, embed_dim_spa, mask_R, HS, HT, num_route):
        super(cap_adj, self).__init__()
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.mask_R = mask_R
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.LRelu = nn.LeakyReLU()

    def forward(self, x, teb):
        batch_size = x.size(0)
        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)
        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)
        Dcaps_in = torch.matmul(squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
                             Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)).permute(0, 1, 3, 4, 2)
        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to('cuda:0')
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b += uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)
        return c

class cap(nn.Module):
    def __init__(self, dim, num_nodes, timesteps, embed_dim, embed_dim_spa, HS, HT, num_route):
        super(cap, self).__init__()
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.t_adj = nn.Parameter(torch.randn(embed_dim_spa, HT, self.TT), requires_grad=True)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.weights_spa = nn.Parameter(torch.FloatTensor(embed_dim, dim, dim))
        self.bias_spa = nn.Parameter(torch.FloatTensor(embed_dim, dim))

        self.LRelu = nn.LeakyReLU()

        mask_template = (torch.linspace(1, timesteps, steps=timesteps)) / 12.
        self.register_buffer('mask_template', mask_template)

    def forward(self, x, node_embeddings, time_eb, teb):
        batch_size = x.size(0)
        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)
        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)
        Dcaps_in = torch.matmul(squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
                             Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)).permute(0, 1, 3, 4, 2)
        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to('cuda:0')
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b += uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)
        # c_return = b + dadj.unsqueeze(-1)

        s = torch.einsum('bthn,btnd->bthd', c.squeeze(-1), Pcaps_out)

        time_index = self.mask_template.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        hyperEmbeds_spa = s + time_index
        hyperEmbeds_spa = hyperEmbeds_spa.reshape(batch_size, -1, self.dim)

        dynamic_adj = torch.einsum('bd,dhk->bhk', time_eb, self.t_adj)
        hyperEmbeds_tem = self.LRelu(torch.einsum('bhk,bkd->bhd', dynamic_adj, hyperEmbeds_spa))
        retEmbeds_tem = self.LRelu(torch.einsum('bkh,bhd->bkd', dynamic_adj.transpose(-1, -2), hyperEmbeds_tem))
        retEmbeds_tem = retEmbeds_tem.reshape(batch_size, self.timesteps, -1, self.dim) + s

        v = squash(retEmbeds_tem)
        reconstruction = torch.einsum('btnh,bthd->btnd', c.squeeze(-1).transpose(-1, -2), v)

        weights_spatial = torch.einsum('nd,dio->nio', node_embeddings, self.weights_spa)
        bias_spatial = torch.matmul(node_embeddings, self.bias_spa)                 #N, dim_out
        out = torch.einsum('btni,nio->btno', reconstruction, weights_spatial) + bias_spatial  # b, N, dim_out

        return self.LRelu(out + x), c.detach(), dynamic_adj.detach()


class hyperTem(nn.Module):
    def __init__(self, timesteps, num_node, dim_in, dim_out, embed_dim, HT_Tem):
        super(hyperTem, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HT_Tem, timesteps), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):

        adj_dynamics = torch.einsum('nk,kht->nht', node_embeddings, self.adj).permute(1, 2, 0)
        hyperEmbeds = torch.einsum('htn,btnd->bhnd', adj_dynamics, eb)
        retEmbeds = torch.einsum('thn,bhnd->btnd', adj_dynamics.transpose(0, 1), hyperEmbeds)

        weights = torch.einsum('btd,dio->btio', time_eb, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(time_eb, self.bias_pool).unsqueeze(2)                       #N, dim_out
        out = torch.einsum('btni,btio->btno', retEmbeds, weights) + bias     #b, N, dim_out
        return self.act(out + eb)

class hyperSpa(nn.Module):
    def __init__(self, num_node, dim_in, dim_out, embed_dim, HS_Spa):
        super(hyperSpa, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HS_Spa, num_node), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):

        adj_dynamics = torch.einsum('btk,khn->bthn', time_eb, self.adj).permute(1, 2, 0)
        hyperEmbeds = self.act(torch.einsum('bthn,btnd->bthd', adj_dynamics, eb))
        retEmbeds = self.act(torch.einsum('btnh,bthd->btnd', adj_dynamics.transpose(-1, -2), hyperEmbeds))

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                     #N, dim_out
        out = torch.einsum('btni,nio->btno', retEmbeds, weights) + bias     #b, N, dim_out
        return self.act(out + eb)


class time_feature(nn.Module):
    def __init__(self, embed_dim):
        super(time_feature, self).__init__()

        self.ln_day = nn.Linear(1, embed_dim)
        self.ln_week = nn.Linear(1, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        day = self.ln_day(eb[:, :, 0:1])
        week = self.ln_week(eb[:, :, 1:2])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb

class time_feature_spg(nn.Module):
    def __init__(self, embed_dim):
        super(time_feature_spg, self).__init__()

        self.ln_day = nn.Linear(12, embed_dim)
        self.ln_week = nn.Linear(12, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        day = self.ln_day(eb[:, :, 0])
        week = self.ln_week(eb[:, :, 1])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb

class STHCN(nn.Module):
    def __init__(self, args):
        super(STHCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.node_embeddings_spg = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)


        self.hyperTem1 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem2 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem3 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem4 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)

        self.time_feature1 = time_feature(self.embed_dim)
        self.time_feature1_ = time_feature(self.embed_dim_spa)
        self.time_feature2 = time_feature_spg(self.embed_dim_spa)

        self.cap1 = cap(self.hidden_dim, args.num_nodes, self.horizon, self.embed_dim, self.embed_dim_spa, self.HS, self.HT, self.num_route)
        self.cap2 = cap(self.hidden_dim, args.num_nodes, self.horizon, self.embed_dim, self.embed_dim_spa, self.HS, self.HT, self.num_route)

    def forward(self, source, x_in):
        #source: B, T_1, N, D

        day_index = source[:, :, 0, self.input_base_dim:self.input_base_dim+1]
        week_index = source[:, :, 0, self.input_base_dim+1:self.input_base_dim+2]

        time_eb = self.time_feature1(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        teb = self.time_feature1_(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        time_eb_spg = self.time_feature2(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)

        # print(time_eb.shape, teb.shape, time_eb_spg.shape)

        xt1 = self.hyperTem1(x_in, self.node_embeddings, time_eb)
        x_hyperTem_gnn1, HS1, HT1 = self.cap1(xt1, self.node_embeddings_spg, time_eb_spg, teb)
        xt2 = self.hyperTem2(x_hyperTem_gnn1, self.node_embeddings, time_eb)

        xt3 = self.hyperTem3(xt2, self.node_embeddings, time_eb)
        x_hyperTem_gnn3, HS3, HT3 = self.cap2(xt3, self.node_embeddings_spg, time_eb_spg, teb)
        xt4 = self.hyperTem4(x_hyperTem_gnn3, self.node_embeddings, time_eb)

        return xt4, HS1, HS3


class Hypergraph_encoder(nn.Module):
    def __init__(self, args):
        super(Hypergraph_encoder, self).__init__()
        self.device = args.device
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.lag
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode
        self.scaler_zeros = args.scaler_zeros
        self.interval = args.interval
        self.week_day = args.week_day
        self.mask_ratio = args.mask_ratio
        self.ada_mask_ratio = args.ada_mask_ratio
        self.ada_type = args.ada_type
        self.change_epoch = args.change_epoch
        self.epochs = args.epochs

        self.dim_in_flow = nn.Linear(self.input_base_dim, self.hidden_dim, bias=True)

        self.STHCN_encode = STHCN(args)
        self.hyperguide1 = torch.randn(self.hidden_dim, self.horizon, self.HS, self.num_node).to('cuda:0')
        self.MLP_RL = MLP_RL(args.input_base_dim, self.HS, self.hidden_dim, self.embed_dim, self.device)
        self.teb4mask = time_feature(self.embed_dim)
        self.neb4mask = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.act = nn.LeakyReLU()

    def forward(self, source, label, epoch=None):
        if self.mode == 'pretrain':
            if epoch <= self.change_epoch:
                # random mask sta2
                mask_random_init = torch.rand_like(source[..., 0:self.input_base_dim].reshape(-1)).to('cuda:0')
                _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)
                mask_num = int(mask_random_init.shape[0] * self.mask_ratio)
                max_idx = max_idx_random[:mask_num]  # NYC_TAXI
                mask_random = torch.ones_like(max_idx_random)
                mask_random = mask_random.scatter_(0, max_idx, 0)
                mask_random = mask_random.reshape(-1, self.horizon, self.num_node, self.input_base_dim)
                final_mask = mask_random

                # get the HS first
                day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
                week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
                time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
                guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

                # get the classification label
                softmax_guide_weight = F.softmax(guide_weight, dim=-1)
            else:
                ### intra-class  inter-class

                # get the HS first
                day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
                week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
                time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
                guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

                # get the classification label
                softmax_guide_weight = F.softmax(guide_weight, dim=-1)
                max_value, max_idx_all = torch.sort(softmax_guide_weight, dim=-1, descending=True)
                label_c = max_idx_all[..., 0]    # [batch_size, time_steps, num_node]

                # calculate number of random mask and adaptive mask
                train_process = ((epoch - self.change_epoch) / (self.epochs - self.change_epoch)) * self.ada_mask_ratio
                if train_process > 1 :
                    train_process = 1
                mask_num_sum = int(source[:, :, :, 0].reshape(-1).shape[0] * self.mask_ratio)
                adaptive_mask_num = int(mask_num_sum * train_process)
                random_mask_num = mask_num_sum - adaptive_mask_num

                ### adaptive mask
                # random choose mask class until the adaptive_mask_num<=select_num
                list_c = list(range(0, self.HS))
                random.shuffle(list_c)
                select_c = torch.zeros_like(label_c).to(self.device)
                select_d = torch.zeros_like(label_c).to(self.device)
                select_f = torch.zeros_like(label_c).to(self.device)
                select_num = 0
                i = 0

                if self.ada_type == 'all':
                    while select_num < adaptive_mask_num:
                        select_c[label_c == list_c[i]] = 1
                        select_num = torch.sum(select_c)
                        i = i + 1
                    if i >= 2:
                        for k in range(i-1):
                            select_d[label_c == list_c[k]] = 1
                            adaptive_dnum = torch.sum(select_d)
                        select_f[label_c == list_c[i-1]] = 1
                    else:
                        adaptive_dnum = 0
                        select_f = select_c.clone()
                else:
                    while select_num < adaptive_mask_num:
                        select_c[label_c == list_c[i]] = 1
                        select_num = torch.sum(select_c)
                        i = i + 1
                    adaptive_dnum = 0
                    select_f = select_c.clone()

                # randomly choose top adaptive_mask_num to mask
                select_f = select_f.reshape(-1, self.horizon*self.num_node).reshape(-1)
                select_d = select_d.reshape(-1, self.horizon*self.num_node).reshape(-1)
                mask_adaptive_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda:0')
                mask_adaptive_init = select_f * mask_adaptive_init
                _, max_idx_adaptive = torch.sort(mask_adaptive_init, dim=0, descending=True)

                select_idx_adaptive = max_idx_adaptive[:(adaptive_mask_num-adaptive_dnum)]

                mask_adaptive = torch.ones_like(max_idx_adaptive)
                mask_adaptive = mask_adaptive.scatter_(0, select_idx_adaptive, 0)
                mask_adaptive = mask_adaptive * (1-select_d)

                # random mask
                mask_random_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda:0')
                mask_random_init = mask_adaptive * mask_random_init
                _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)

                select_idx_random = max_idx_random[:random_mask_num]
                mask_random = torch.ones_like(max_idx_random)
                mask_random = mask_random.scatter_(0, select_idx_random, 0)
                mask_random = mask_random.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)

                # final_mask
                mask_adaptive = mask_adaptive.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)
                final_mask = (mask_adaptive * mask_random).unsqueeze(-1)
                if self.input_base_dim != 1:
                    final_mask = final_mask.repeat(1, 1, 1, self.input_base_dim)

            final_mask = final_mask.detach()
            mask_source = final_mask * source[..., 0:self.input_base_dim]
            mask_source[final_mask==0] = self.scaler_zeros
            x_flow_eb = self.dim_in_flow(mask_source)
        else:
            x_flow_eb = self.dim_in_flow(source[..., 0:self.input_base_dim])
        x_flow_encode, HS1, HS2 = self.STHCN_encode(source, x_flow_eb)

        if self.mode == 'pretrain':
            HS_cat = HS1.squeeze(-1).transpose(-1, -2)
            return x_flow_encode, final_mask[..., :self.input_base_dim], softmax_guide_weight, HS_cat
        else:
            return x_flow_encode

class Hypergraph_decoder(nn.Module):
    def __init__(self, args):
        super(Hypergraph_decoder, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode

        self.time_feature1_ = time_feature(self.embed_dim_spa)
        self.time_feature2_ = time_feature(self.embed_dim_spa)

        self.STHCN_decode = STHCN(args)
        self.dim_flow_out = nn.Linear(self.hidden_dim, self.input_base_dim, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, source, flow_encode_eb):
        flow_decode, HS1, HS2 = self.STHCN_decode(source, flow_encode_eb)
        flow_out = self.dim_flow_out(flow_decode)
        return flow_out, flow_decode


class GPTST_Model(nn.Module):
    def __init__(self, args):
        super(GPTST_Model, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode
        self.model = args.model

        self.encoder = Hypergraph_encoder(args)
        self.decoder = Hypergraph_decoder(args)

    def forward_pretrain(self, source, label, batch_seen=None, epoch=None):
        flow_encode_eb, mask, probability, HS1 = self.encoder(source, label, epoch)
        flow_out, flow_decode = self.decoder(source, flow_encode_eb)
        return flow_out, flow_decode, 1-mask, probability, HS1

    def forward_fune(self, source, label):
        flow_encode_eb = self.encoder(source, label)
        return flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb

    def forward(self, source, label, batch_seen=None, epoch=None):
        if self.mode == 'pretrain':
            return self.forward_pretrain(source, label, batch_seen, epoch)
        else:
            return self.forward_fune(source, label)