import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree


def sep_data(dataset, seed=0, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [data['y'] for data in dataset]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    return idx_list[fold_idx]
    

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)
            
            
class Indegree(object):
    r"""Adds the globally normalized node degree to the node features.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    def __init__(self, norm=False, max_value=None):
        self.norm = norm
        self.max = max_value

    def __call__(self, data):
        col, x = data.edge_index[1], data.x
        deg = degree(col, data.num_nodes)
        #print(deg)
        if self.norm:
            deg = deg / (deg.max() if self.max is None else self.max)
        deg = deg.view(-1, 1)
        if x is None:
            data.x = deg
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__, self.norm, self.max)
