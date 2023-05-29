import MDAnalysis as mda
from MDAnalysis.coordinates import DCD
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
atom_type_mapping = {"CA": 0,"CB":1,"C": 2, "N": 3, "O":4, "OXT":5,"H2":6,"H3":6,"HB3":6,"HB2":6,"HB1":6,"HA":6,"H":7}  # Add other atom types if necessary
edge_type_mapping = {"CA-CB": 0,"CA-C":1,"C-N": 2,"C-O": 3,"N-CA": 4,"OXT-C": 5,"CA-H": 6,"CB-H": 7,"N-H": 8}
n_atom_types = len(atom_type_mapping)
n_edge_types = len(edge_type_mapping)

class GraphNodeFeature(torch.nn.module):
    def __init__(self,dim):
        super(GraphNodeFeature, self).__init__()
        self.embedding = torch.nn.Embedding(n_atom_types, dim)
        self.degree_encoder = torch.nn.Embedding(4, dim)
    def forward(self, batched_data):
        x, degree = batched_data['x'], batched_data['degree']
        x = self.embedding(x)
        degree = self.degree_encoder(degree)
        node_feature = x + degree
        return node_feature
class GraphAttnBias(torch.nn.Module):
    def __init__(self,heads,num_edge_dis):
        super(GraphAttnBias, self).__init__()
        self.Spatial_Encoding = torch.nn.Embedding(n_atom_types, heads)
        '''
        It's hard for the big graph that has many edges.No egde_feature.
        '''
        # self.Edge_Encoding = torch.nn.Embedding(n_edge_types, heads,padding_idx= -1)
        # self.edge_dis_encoder = torch.nn.Embedding(
        #         num_edge_dis * heads * heads, 1
        #     )
    def forward(self, x):
        x = self.Spatial_Encoding(x)
        return x

    
class self_attention_with_bias(torch.nn.Module):
    def __init__(self, d_model, n_head ):
        super(self_attention_with_bias, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model//n_head
        self.w_qs = torch.nn.Linear(d_model, d_model)
        self.w_ks = torch.nn.Linear(d_model, d_model)
        self.w_vs = torch.nn.Linear(d_model, d_model)

    def forward(self, x, bias, mask=None):
        bs = x.size(0)
        q = self.w_qs(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_ks(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        atten_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias
        if mask is not None:
            atten_logits = atten_logits.masked_fill(mask == 0, -1e9)
        atten_weights = torch.softmax(atten_logits, dim=-1)
        v = self.w_vs(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        output = torch.matmul(atten_weights, v)
        return output,atten_weights

class Graph_attention(torch.nn.Module):
    def __init__(self, d_model, n_head ):
        super(Graph_attention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model//n_head
        self.w_qs = torch.nn.Linear(d_model, d_model)
        self.w_ks = torch.nn.Linear(d_model, d_model)
        self.w_vs = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, 1)
    def forward(self, x, r,bias, mask=None):
        bs = x.size(0)
        q = self.w_qs(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_ks(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        atten_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias
        if mask is not None:
            atten_logits = atten_logits.masked_fill(mask == 0, -1e9)
        atten_weights = torch.softmax(atten_logits, dim=-1)
        atten_3D = r.unsqueeze(1)*atten_weights.unsqueeze(-1).permute(0, 1, 4, 2, 3)
        v = self.w_vs(x).view(bs, -1, self.n_head, self.head_dim).transpose(1, 2).unsqueeze(2)
        o = torch.matmul(atten_3D, v).permute(0, 3, 2, 1, 4).view(bs, -1, 3,self.d_model) 
        force = self.o_proj(o)
        return force

class Encoder_Layer(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):   
        super(Encoder_Layer, self).__init__()
        self.self_attn = self_attention_with_bias(d_model, n_head)
        self.dropout = torch.nn.Dropout(dropout)
        self.pre_norm1 = torch.nn.LayerNorm(d_model)
        self.pre_norm2 = torch.nn.LayerNorm(d_model)
        self.linear1 = torch.nn.Linear(d_model, d_model)
        self.linear2 = torch.nn.Linear(d_model, d_model)
    def forward(self, x, bias, mask=None):
        res = x
        x = self.pre_norm1(x)
        x, _ = self.self_attn(x, bias, mask)
        x = self.dropout(x)
        x = x + res
        res = x
        x = self.pre_norm2(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Graph_3D(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(Graph_3D, self).__init__()
        self.self_attn = Graph_attention(d_model, n_head)
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

    def forward(self, r,x, mask=None):
        r0 = torch.cdist(r, r, p=2)
        bias = torch.exp(-r0)
        delta_pos = (r.unsqueeze(1) - r.unsqueeze(2))/(r0+1e-6)
        graph_output = self.self_attn(x, delta_pos, bias, mask)
        return graph_output



class Graphormer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.1):
        super(Graphormer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers

        # List of encoder layers
        self.encoders = nn.ModuleList([Encoder_Layer(d_model, n_head, dropout) for _ in range(num_layers)])

        # 3D graph layer
        self.graph_3d = Graph_3D(d_model, n_head, dropout)

    def forward(self, x, r, mask=None):
        # Pass through each EncoderLayer
        for i in range(self.num_layers):
            x = self.encoders[i](x, mask)

        # Pass through the 3D graph layer
        x = self.graph_3d(r, x, mask)

        return x
        
