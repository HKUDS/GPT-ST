import torch
import torch.nn as nn
from Pretrain_model.GPTST import GPTST_Model

class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.HS_fc = nn.Linear(dim, dim, bias=True)
        self.HT_fc = nn.Linear(dim, dim, bias=True)
        self.output_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, flow_eb, time_eb):
        XS = self.HS_fc(flow_eb)
        XT = self.HT_fc(time_eb)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, flow_eb), torch.multiply(1 - z, time_eb))
        H = self.output_fc(H)
        return H

class Enhance_model(nn.Module):
    def __init__(self, args, args_predictor):
        super(Enhance_model, self).__init__()
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
        self.load_pretrain_path = args.load_pretrain_path
        self.log_dir = args.log_dir

        if self.mode == 'eval':
            self.pretrain_model = GPTST_Model(args)
            self.load_pretrained_model()
            self.fusion = Fusion(self.hidden_dim)
            self.lin_test = nn.Linear(self.input_base_dim, self.hidden_dim)

        if self.mode == 'ori':
            dim_in = self.input_base_dim
        else:
            dim_in = self.hidden_dim
        dim_out = self.output_dim

        if self.model == 'MTGNN':
            from MTGNN.MTGNN import MTGNN
            self.predictor = MTGNN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STGCN':
            from STGCN.stgcn import STGCN
            self.predictor = STGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STSGCN':
            from STSGCN.STSGCN import STSGCN
            self.predictor = STSGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'ASTGCN':
            from ASTGCN.ASTGCN import ASTGCN
            self.predictor = ASTGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'GWN':
            from GWN.GWN import GWNET
            self.predictor = GWNET(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'TGCN':
            from TGCN.TGCN import TGCN
            self.predictor = TGCN(args_predictor, args.device, dim_in)
        elif self.model == 'STFGNN':
            from STFGNN.STFGNN import STFGNN
            self.predictor = STFGNN(args_predictor, dim_in)
        elif self.model == 'STGODE':
            from STGODE.STGODE import ODEGCN
            self.predictor = ODEGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STWA':
            from ST_WA.ST_WA import STWA
            self.predictor = STWA(args_predictor, args.device, dim_in)
        elif self.model == 'MSDR':
            from MSDR.gmsdr_model import GMSDRModel
            args_predictor.input_dim = dim_in
            self.predictor = GMSDRModel(args_predictor, args.device)
        elif self.model == 'DMVSTNET':
            from DMVSTNET_demand.DMVSTNET import DMVSTNet
            self.predictor = DMVSTNet(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'CCRNN':
            from CCRNN_demand.CCRNN import EvoNN2
            self.predictor = EvoNN2(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STMGCN':
            from STMGCN_demand.STMGCN import ST_MGCN
            self.predictor = ST_MGCN(args_predictor, args.device, dim_in, dim_out)
        else:
            raise ValueError

    def load_pretrained_model(self):
        self.pretrain_model.load_state_dict(torch.load(self.log_dir + self.load_pretrain_path))
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def forward(self, source, label, batch_seen=None):
        if self.mode == 'ori':
            return self.forward_ori(source, label, batch_seen)
        else:
            return self.forward_pretrain(source, label, batch_seen)

    def forward_pretrain(self, source, label, batch_seen=None):
        x_pretrain_flow, _, _, _, _ = self.pretrain_model(source, label)
        x_t1 = self.lin_test(source[..., :self.input_base_dim])
        pretrain_eb = self.fusion(x_pretrain_flow, x_t1)
        if self.model == 'CCRNN':
            if label is None:
                x_predic = self.predictor(pretrain_eb, None, None)
            else:
                x_predic = self.predictor(pretrain_eb, label[:, :, :, 0:self.input_base_dim], None)
        else:
            x_predic = self.predictor(pretrain_eb)
        return x_predic, x_predic, x_predic, x_predic, x_predic

    def forward_ori(self, source, label=None, batch_seen=None):
        if self.model == 'CCRNN':
            if label is None:
                x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim], None, None)
            else:
                x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim], label[:, :, :, 0:self.input_base_dim], None)
        else:
            x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim])
        return x_predic, x_predic, x_predic, x_predic, x_predic
