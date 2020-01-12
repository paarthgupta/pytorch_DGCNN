import ctypes
import numpy as np
import os
import sys
import torch
import pdb

class _hgnn_lib(object):

    def __init__(self):
        pass

    def PrepareSparseMatrices(self, S_list):
        full_I = []
        full_J = []
        full_V = []
        I_prefix = 0
        J_prefix = 0
        list_I = []
        list_J = []
        for i, S in enumerate(S_list):
            n_rows, n_cols = S.shape
            I, J, V = scipy.sparse.find(S)
            full_I += list(I+I_prefix)
            full_J += list(J+J_prefix)
            full_V += list(V)
            list_I += [i]*n_rows
            list_J += list(range(I_prefix, I_prefix + n_rows))
            I_prefix += n_rows
            J_prefix += n_cols
        S = torch.sparse.FloatTensor(torch.LongTensor([full_I, full_J]), torch.FloatTensor(full_V), torch.Size([I_prefix, J_prefix]))
        list_nodes_matrix = torch.sparse.FloatTensor(torch.LongTensor([list_I, list_J]), torch.LongTensor([1]*len(list_I)), torch.Size([len(S_list), I_prefix]))
        return S, list_nodes_matrix

HGNNLIB = _hgnn_lib()

