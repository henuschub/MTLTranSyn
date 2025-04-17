import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout, Softmax
from static.constant import DrugAE_InputDim, DrugAE_OutputDim, CELLAE_InputDim, CellAE_OutputDim, MTLTranSyn_InputDim
from utils.tools import init_weights
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.container import ModuleList


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class DrugAE(Module):
    def __init__(self, input_dim=DrugAE_InputDim, output_dim=DrugAE_OutputDim):
        super(DrugAE, self).__init__()
        if output_dim == 32 or output_dim == 64:
            hidden_dim = 256
        elif output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        elif output_dim == 512:
            hidden_dim = 1024
        else:
            hidden_dim = 4096
        self.encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim)
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class CellLineAE(Module):
    def __init__(self, input_dim=CELLAE_InputDim, output_dim=CellAE_OutputDim):
        super(CellLineAE, self).__init__()
        if output_dim == 32 or output_dim == 64:
            hidden_dim = 256
        elif output_dim == 128 or output_dim == 256:
            hidden_dim = 512
        elif output_dim == 512:
            hidden_dim = 1024
        else:
            hidden_dim = 4096
        self.encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim)
        init_weights(self._modules)

    def forward(self, input):
        x = self.encoder(input)
        y = self.decoder(x)
        return y


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        bert_n_heads = 3
        drop_out_rating = 0.3

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.att1 = EncoderLayer(hidden_dim, bert_n_heads)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim * 2)
        self.bn2 = torch.nn.BatchNorm1d(output_dim * 2)
        self.att2 = EncoderLayer(output_dim * 2, bert_n_heads)
        self.l3 = torch.nn.Linear(output_dim * 2, output_dim)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.att1(X)
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.att2(X)
        X = self.l3(X)
        return X


class FraEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraEncoder, self).__init__()
        between_hid_final = (hidden_dim + output_dim) // 2
        bert_n_heads = 3
        drop_out_rating = 0.3
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.attn1 = EncoderLayer(hidden_dim, bert_n_heads)
        self.l2 = torch.nn.Linear(hidden_dim, between_hid_final)
        self.bn2 = torch.nn.BatchNorm1d(between_hid_final)
        self.attn2 = EncoderLayer(between_hid_final, bert_n_heads)
        self.l3 = torch.nn.Linear(between_hid_final, output_dim)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.attn1(X)

        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.attn2(X)

        X = self.l3(X)

        return X


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        bert_n_heads = 3
        drop_out_rating = 0.3
        self.l1 = torch.nn.Linear(output_dim, output_dim * 2)
        self.bn1 = torch.nn.BatchNorm1d(output_dim * 2)
        self.att1 = EncoderLayer(output_dim * 2, bert_n_heads)
        self.l2 = torch.nn.Linear(output_dim * 2, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.att2 = EncoderLayer(hidden_dim, bert_n_heads)
        self.l3 = torch.nn.Linear(hidden_dim, input_dim)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.att1(X)
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.att2(X)
        X = self.l3(X)

        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)
        return X


class CrossAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(CrossAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.W_Q1 = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K1 = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V1 = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
        self.fc1 = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X, X1):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        Q1 = self.W_Q1(X1).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K1 = self.W_K1(X1).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V1 = self.W_V1(X1).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q1, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores1 = torch.matmul(Q, K1.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        attn1 = torch.nn.Softmax(dim=-1)(scores1)
        context = torch.matmul(attn, V)
        context1 = torch.matmul(attn1, V1)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        context1 = context1.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        output1 = self.fc1(context1)
        return output, output1


class EncoderLayertwo(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayertwo, self).__init__()
        self.attn = CrossAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)
        self.AN1_1 = torch.nn.LayerNorm(input_dim)
        self.l1_1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2_1 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, X1):
        output, output1 = self.attn(X, X1)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)
        X1 = self.AN1_1(output1 + X1)
        output1 = self.l1_1(X1)
        X1 = self.AN2_1(output1 + X1)

        return X, X1


class SelfAttUnit(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SelfAttUnit, self).__init__()
        bert_n_heads = 3
        drop_out_rating = 0.3

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.att1 = EncoderLayer(hidden_dim, bert_n_heads)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim * 2)
        self.bn2 = torch.nn.BatchNorm1d(output_dim * 2)
        self.att2 = EncoderLayer(output_dim * 2, bert_n_heads)
        self.l3 = torch.nn.Linear(output_dim * 2, output_dim)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.att1(X)
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        X = self.att2(X)
        X = self.l3(X)

        return X


class SelfAttnExpert(nn.Module):
    def __init__(self, input_size_x, shared_input_size, den_hidden_size, final_output_size):
        super(SelfAttnExpert, self).__init__()
        input_dim = input_size_x + shared_input_size
        hidden_dim = den_hidden_size
        output_dim = final_output_size
        self.encoder = FraEncoder(input_dim, hidden_dim, output_dim)

    def forward(self, d, c):
        X = torch.cat((d, c), dim=1)
        X = self.encoder(X)
        return X


class DualAttnExpert(nn.Module):
    def __init__(self, input_size_x, shared_input_size, den_hidden_size, final_output_size):
        super(DualAttnExpert, self).__init__()
        input_dim = input_size_x
        hidden_dim = den_hidden_size
        output_dim = final_output_size
        self.encoder = FraEncoder(input_dim, hidden_dim, output_dim)
        self.l5 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, d, c):
        X1 = self.encoder(d)
        X2 = self.encoder(c)
        X = torch.cat((X1, X2), dim=1)
        X = self.l5(X)
        return X


class CrossAttnExpert(torch.nn.Module):
    def __init__(self, input_size_x, shared_input_size, den_hidden_size, final_output_size):
        super(CrossAttnExpert, self).__init__()

        half_size = input_size_x
        len_after_AE = final_output_size
        bert_n_heads = 3
        drop_out_rating = 0.3
        self.l1 = torch.nn.Linear(half_size, len_after_AE * 2)
        self.bn1 = torch.nn.BatchNorm1d(len_after_AE * 2)
        self.att1 = EncoderLayer(len_after_AE * 2, bert_n_heads)
        self.l2 = torch.nn.Linear(half_size, len_after_AE * 2)
        self.bn2 = torch.nn.BatchNorm1d(len_after_AE * 2)
        self.att2 = EncoderLayer(len_after_AE * 2, bert_n_heads)
        self.l3 = torch.nn.Linear(len_after_AE * 2, len_after_AE)
        self.bn3 = torch.nn.BatchNorm1d(len_after_AE)
        self.l4 = torch.nn.Linear(len_after_AE * 2, len_after_AE)
        self.bn4 = torch.nn.BatchNorm1d(len_after_AE)
        self.att3 = EncoderLayertwo(len_after_AE, bert_n_heads)
        self.l5 = torch.nn.Linear(len_after_AE * 2, len_after_AE)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, d, c):
        X1 = d
        X2 = c
        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att1(X1)
        X2 = self.dr(self.bn2(self.ac(self.l2(X2))))
        X2 = self.att2(X2)
        X1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X2 = self.dr(self.bn4(self.ac(self.l4(X2))))
        X1, X2 = self.att3(X1, X2)
        X = torch.cat((X1, X2), dim=1)
        X = self.l5(X)
        return X


class MTLTranSyn(Module):
    def __init__(self, hidden_neurons, input_dim=MTLTranSyn_InputDim):
        super(MTLTranSyn, self).__init__()
        self.config = {
            'input_size_x': 128,
            'shared_input_size': 128,
            'den_hidden_size': hidden_neurons[1],
            'final_output_size': 256,
            'num_experts': 6,
            'num_experts_per_type': 2,
            'gate_input_size': 512,
            'experts_hidden': hidden_neurons[1],
        }
        self.selfattn_experts = nn.ModuleList(
            [SelfAttnExpert(self.config['input_size_x'], self.config['shared_input_size'],
                            self.config['den_hidden_size'], self.config['final_output_size'])
             for _ in range(self.config['num_experts_per_type'])]
        )
        self.dualattn_experts = nn.ModuleList(
            [DualAttnExpert(self.config['input_size_x'], self.config['shared_input_size'],
                            self.config['den_hidden_size'], self.config['final_output_size'])
             for _ in range(self.config['num_experts_per_type'])]
        )
        self.crossattn_experts = nn.ModuleList(
            [CrossAttnExpert(self.config['input_size_x'], self.config['shared_input_size'],
                             self.config['den_hidden_size'], self.config['final_output_size'])
             for _ in range(self.config['num_experts_per_type'])]
        )
        self.w_gates_unli = SelfAttUnit(self.config['gate_input_size'], self.config['num_experts'],
                                        self.config['experts_hidden'])
        self.w_gates_unli1 = SelfAttUnit(self.config['gate_input_size'], self.config['num_experts'],
                                         self.config['experts_hidden'])
        self.fc_layer = Sequential(
            Linear(input_dim, hidden_neurons[0]),
            BatchNorm1d(hidden_neurons[0]),
            ReLU(True),
            Linear(hidden_neurons[0], hidden_neurons[1]),
            ReLU(True)
        )
        self.synergy_out_1 = Sequential(
            Linear(2 * hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True),
            Linear(128, 1)
        )
        self.synergy_out_2 = Sequential(
            Linear(2 * hidden_neurons[1], hidden_neurons[2]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[2], 128),
            ReLU(True),
            Linear(128, 2),
            Softmax(dim=1)
        )
        self.sensitivity_out_1 = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 64),
            ReLU(True),
            Linear(64, 1)
        )
        self.sensitivity_out_2 = Sequential(
            Linear(hidden_neurons[1], hidden_neurons[3]),
            ReLU(True),
            Dropout(0.5),
            Linear(hidden_neurons[3], 64),
            ReLU(True),
            Linear(64, 2),
            Softmax(dim=1)
        )
        init_weights(self._modules)

    def forward(self, d1, d2, c_exp):
        selfattn_experts_o1 = [e(d1, c_exp) for e in self.selfattn_experts]
        dualattn_experts_o2 = [e(d1, c_exp) for e in self.dualattn_experts]
        crossattn_experts_o3 = [e(d1, c_exp) for e in self.crossattn_experts]
        selfattn_experts_o11 = [e(d2, c_exp) for e in self.selfattn_experts]
        dualattn_experts_o22 = [e(d2, c_exp) for e in self.dualattn_experts]
        crossattn_experts_o33 = [e(d2, c_exp) for e in self.crossattn_experts]
        all_experts_o = selfattn_experts_o1 + dualattn_experts_o2 + crossattn_experts_o3
        all_experts_o1 = selfattn_experts_o11 + dualattn_experts_o22 + crossattn_experts_o33
        experts_o_tensor = torch.stack(all_experts_o)
        experts_o_tensor1 = torch.stack(all_experts_o1)
        gateinput = torch.cat((d1, c_exp, d2, c_exp), dim=1)
        gates_o = self.softmax(self.w_gates_unli(gateinput))
        gates_o1 = self.softmax(self.w_gates_unli1(gateinput))
        tower_input1 = torch.sum(gates_o.t().unsqueeze(2).expand(-1, -1, self.final_output_size) * experts_o_tensor,
                                 dim=0)
        tower_input2 = torch.sum(
            gates_o1.t().unsqueeze(2).expand(-1, -1, self.final_output_size) * experts_o_tensor1, dim=0)
        tower_input1 = self.fc_layer(tower_input1)
        tower_input2 = self.fc_layer(tower_input2)

        d1_sen = tower_input1
        syn = torch.cat((tower_input1, tower_input2), 1)

        syn_out_1 = self.synergy_out_1(syn)
        syn_out_2 = self.synergy_out_2(syn)
        d1_sen_out_1 = self.sensitivity_out_1(d1_sen)
        d1_sen_out_2 = self.sensitivity_out_2(d1_sen)
        return syn_out_1.squeeze(-1), d1_sen_out_1.squeeze(-1), syn_out_2, d1_sen_out_2
