import math
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn
from abc import ABCMeta, abstractmethod

import models.modules as modules
import models.functions as fn
import numpy as np


__all__ = ['OpticalMolGen_RNN']

class MoleculeGenerator(nn.Block):

    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 *args, **kwargs):
        super(MoleculeGenerator, self).__init__()
        self.N_A = N_A
        self.N_B = N_B
        self.D = D
        self.F_e = F_e
        self.F_skip = F_skip
        self.F_c = list(F_c) if isinstance(F_c, tuple) else F_c
        self.Fh_policy = Fh_policy
        self.activation = fn.get_activation(activation)

        with self.name_scope():
            # embeddings
            self.embedding_atom = nn.Embedding(self.N_A, self.F_e)
            self.embedding_mask = nn.Embedding(3, self.F_e)
            # graph conv
            self._build_graph_conv(*args, **kwargs)

            # fully connected
            self.dense = nn.Sequential()
            for i, (f_in, f_out) in enumerate(zip([self.F_skip, ] + self.F_c[:-1], self.F_c)):
                self.dense.add(modules.Linear_BN(f_in, f_out))

            # policy
            self.policy_0 = self.params.get('policy_0', shape=[self.N_A, ],
                                            init=mx.init.Zero(),
                                            allow_deferred_init=False)
            self.policy_h = modules.Policy(self.F_c[-1], self.Fh_policy, self.N_A, self.N_B)

        self.mode = 'loss'

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_conv_forward(self, X, A):
        raise NotImplementedError

    def _policy_0(self, ctx):
        policy_0 = nd.exp(self.policy_0.data(ctx)) # softmax
#         policy_0 = nd.exp(self.policy_0.data(ctx)-nd.max(self.policy_0.data(ctx))) ### stable softmax
        policy_0 = policy_0/policy_0.sum()
        return policy_0

    def _policy(self, X, A, NX, NX_rep, last_append_mask):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep)

        return append, connect, end

    def _likelihood(self, init, append, connect, end,
                    action_0, actions, iw_ids, log_p_sigma,
                    batch_size, iw_size):

        # decompose action:
        action_type, node_type, edge_type, append_pos, connect_pos = \
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4]
        _log_mask = lambda _x, _mask: _mask * nd.log(_x + 1e-10) + (1- _mask) * nd.zeros_like(_x)

        # init
        init = init.reshape([batch_size * iw_size, self.N_A])
        index = nd.stack(nd.arange(action_0.shape[0], ctx=action_0.context, dtype='int32'), action_0, axis=0)
        loss_init = nd.log(nd.gather_nd(init, index) + 1e-10)

        # end
        loss_end = _log_mask(end, nd.cast(action_type == 2, 'float32'))
        # append
        index = nd.stack(append_pos, node_type, edge_type, axis=0)
        loss_append = _log_mask(nd.gather_nd(append, index), nd.cast(action_type == 0, 'float32'))
        # connect
        index = nd.stack(connect_pos, edge_type, axis=0)
        loss_connect = _log_mask(nd.gather_nd(connect, index), nd.cast(action_type == 1, 'float32'))
        # sum up results
        log_p_x = loss_end + loss_append + loss_connect
        log_p_x = fn.squeeze(fn.SegmentSumFn(iw_ids, batch_size*iw_size)(fn.unsqueeze(log_p_x, -1)), -1)
        log_p_x = log_p_x + loss_init

        # reshape
        log_p_x = log_p_x.reshape([batch_size, iw_size])
        log_p_sigma = log_p_sigma.reshape([batch_size, iw_size])
        l = log_p_x - log_p_sigma
        l = fn.logsumexp(l, axis=1) - math.log(float(iw_size))
        if math.isnan(l.mean().asnumpy()[0]):
            print("init",loss_init.mean())
            print("append",loss_append.mean())
            print("connect",loss_connect.mean())
            print("end",loss_end.mean())
            print("log_p",log_p_x)
            print("logsumexp",l)
        return l

    def forward(self, *input):
        if self.mode=='loss' or self.mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p, \
            batch_size, iw_size = input

            init = self._policy_0(X.context).tile([batch_size * iw_size, 1])
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size)
            if self.mode=='likelihood':
                return l
            else:
                return -l.mean()
        elif self.mode == 'decode_0':
            return self._policy_0(input[0])
        elif self.mode == 'decode_step':
            X, A, NX, NX_rep, last_append_mask = input
            return self._policy(X, A, NX, NX_rep, last_append_mask)


class MoleculeGenerator_RNN(MoleculeGenerator):

    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                 N_rnn, *args, **kwargs):
        super(MoleculeGenerator_RNN, self).__init__(N_A, N_B, D, F_e, F_skip, F_c, Fh_policy, activation,
                                                    *args, **kwargs)
        self.N_rnn = N_rnn

        with self.name_scope():
            self.rnn = gluon.rnn.GRU(hidden_size=self.F_c[-1],
                                     num_layers=self.N_rnn,
                                     layout='NTC', input_size=self.F_c[-1] * 2)

    def _rnn_train(self, X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum):
        X_avg = fn.SegmentSumFn(NX_rep, NX.shape[0])(X) / nd.cast(fn.unsqueeze(NX, 1), 'float32')
        X_curr = nd.take(X, indices=NX_cum-1)
        X = nd.concat(X_avg, X_curr, dim=1)

        # rnn
        X = nd.take(X, indices=graph_to_rnn) # batch_size, iw_size, length, num_features
        batch_size, iw_size, length, num_features = X.shape
        X = X.reshape([batch_size*iw_size, length, num_features])
        X = self.rnn(X)

        X = X.reshape([batch_size, iw_size, length, -1])
        X = nd.gather_nd(X, indices=rnn_to_graph)

        return X

    def _rnn_test(self, X, NX, NX_rep, NX_cum, h):
        # note: one partition for one molecule
        X_avg = fn.SegmentSumFn(NX_rep, NX.shape[0])(X) / nd.cast(fn.unsqueeze(NX, 1), 'float32')
        X_curr = nd.take(X, indices=NX_cum - 1)
        X = nd.concat(X_avg, X_curr, dim=1) # size: [NX, F_in * 2]

        # rnn
        X = fn.unsqueeze(X, axis=1)
        X, h = self.rnn(X, h)

        X = fn.squeeze(X, axis=1)
        return X, h

    def _policy(self, X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # rnn
        X_mol = self._rnn_train(X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)

        return append, connect, end

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # linear
        X = self.dense(X)

        # rnn
        X_mol, h = self._rnn_test(X, NX, NX_rep, NX_cum, h)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        
        return append, connect, end, h

    def forward(self, *input):
        if self.mode=='loss' or self.mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p, \
            batch_size, iw_size, \
            graph_to_rnn, rnn_to_graph, NX_cum = input

            init = self._policy_0(X.context).tile([batch_size * iw_size, 1])
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask, graph_to_rnn, rnn_to_graph, NX_cum)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size)
            if self.mode=='likelihood':
                return l
            else:
                return -l.mean()
        elif self.mode == 'decode_0':
            return self._policy_0(input[0])
        elif self.mode == 'decode_step':
            X, A, NX, NX_rep, last_append_mask, NX_cum, h = input
            return self._decode_step(X, A, NX, NX_rep, last_append_mask, NX_cum, h)
        else:
            raise ValueError


class _TwoLayerDense(nn.Block):

    def __init__(self, input_size, hidden_size, output_size):
        super(_TwoLayerDense, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        with self.name_scope():
            self.input = nn.Dense(self.hidden_size, use_bias=False, in_units=self.input_size)
            self.bn_input = modules.BatchNorm(in_channels=self.hidden_size)
            self.output = nn.Dense(self.output_size, use_bias=True, in_units=self.hidden_size)

    def forward(self, c):
        return nd.softmax(self.output(nd.relu(self.bn_input(self.input(c)))), axis=-1)


class CMoleculeGenerator_RNN(MoleculeGenerator_RNN):
    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, N_C, D,
                 F_e, F_skip, F_c, Fh_policy,
                 activation, N_rnn,
                 *args, **kwargs):
        self.N_C = N_C # number of conditional variables
        super(CMoleculeGenerator_RNN, self).__init__(N_A, N_B, D,
                                                     F_e, F_skip, F_c, Fh_policy,
                                                     activation, N_rnn,
                                                     *args, **kwargs)
        with self.name_scope():
            self.dense_policy_0 = _TwoLayerDense(self.N_C, self.N_A * 3, self.N_A)

    @abstractmethod
    def _graph_conv_forward(self, X, A,X_sol,A_sol, c, ids):
        raise NotImplementedError

    def _policy_0(self, c):
        return self.dense_policy_0(c) + 0.0 * self.policy_0.data(c.context)

    def _policy(self, X, A, NX, NX_rep, last_append_mask,
                graph_to_rnn, rnn_to_graph, NX_cum,X_sol,A_sol,NX_sol,c, ids):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        # sol convolution
        D_sol = self._graph_conv_sol_forward(X_sol,A_sol,NX_sol)
        # convolution
        X = self._graph_conv_forward(X, A,NX,NX_rep,D_sol,c, ids)

        # linear
        X = self.dense(X)

        # rnn
        X_mol = self._rnn_train(X, NX, NX_rep, graph_to_rnn, rnn_to_graph, NX_cum)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        #print(append)
        return append, connect, end

    def _decode_step(self, X, A, NX, NX_rep, last_append_mask, NX_cum, h,X_sol,A_sol,NX_sol,  c, ids):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)
        # sol convolution
        D_sol = self._graph_conv_sol_forward(X_sol,A_sol,NX_sol)
        
        # convolution
        X = self._graph_conv_forward(X, A,NX,NX_rep,D_sol,c, ids)
        #print(nd.sum(X))
        # linear
        X = self.dense(X)
        # rnn
        X_mol, h = self._rnn_test(X, NX, NX_rep, NX_cum, h)
        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep, X_mol)
        
        #print("append",append,"connect",connect,"end",end,"h",h)
        
        return append, connect, end, h


    def forward(self, *input):
        if self.mode=='loss' or self.mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p, \
            batch_size, iw_size, \
            graph_to_rnn, rnn_to_graph, NX_cum, \
            X_sol,A_sol,NX_sol,c, ids = input
            init = nd.tile(fn.unsqueeze(self._policy_0(c), axis=1), [1, iw_size, 1])
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask,
                                                    graph_to_rnn, rnn_to_graph, NX_cum,
                                                    X_sol,A_sol,NX_sol, c, ids)
            likelihood = self._likelihood(init, append, connect, end,
                                 action_0, actions, iw_ids, log_p,
                                 batch_size, iw_size)
            if self.mode == 'likelihood':
                return likelihood
            else:
                return -likelihood.mean()
        elif self.mode == 'decode_0':
            return self._policy_0(*input)
        elif self.mode == 'decode_step':
            X, A, NX, NX_rep, last_append_mask, NX_cum, h,X_sol,A_sol,NX_sol, c, ids = input
            return self._decode_step(X, A, NX, NX_rep, last_append_mask, NX_cum, h,X_sol,A_sol,NX_sol, c, ids)
        else:
            raise ValueError

class OpticalMolGen_RNN(CMoleculeGenerator_RNN):

    def __init__(self, N_A, N_B, N_C, D,
                 F_e, F_h, F_skip, F_c, Fh_policy,
                 activation, N_rnn, rename=False):
        self.rename = rename
        super(OpticalMolGen_RNN, self).__init__(N_A, N_B, N_C, D,
                                                 F_e, F_skip, F_c, Fh_policy,
                                                 activation, N_rnn,
                                                 F_h)

    def _build_graph_conv(self, F_h):
        F_G = [43,128,128,128,128,128,256]
        self.F_h = list(F_h) if isinstance(F_h, tuple) else F_h
        self.conv, self.bn, self.conv_sol, self.bn_sol = [], [], [], []
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            conv = modules.GraphConv(f_in, f_out, self.N_B + self.D)
            self.conv.append(conv)
            self.register_child(conv)
            
            if i != 0:
                bn = modules.BatchNorm(in_channels=f_in)
                self.register_child(bn)
            else:
                bn = None
            self.bn.append(bn)
        for i in range(6):
            conv_sol = GraphConvolution(F_G[i+1],F_G[i])
            self.conv_sol.append(conv_sol)
            self.register_child(conv_sol)
            
        self.bn_skip = modules.BatchNorm(in_channels=sum(self.F_h))
        self.linear_skip = modules.Linear_BN(sum(self.F_h), self.F_skip)
        
        # projectors for conditional variable
        self.linear_c = []
        self.linear_sol = []
        self.linear_sol_first = nn.Dense(5,use_bias=False, in_units=256)
        self.register_child(self.linear_sol_first)
        for i, f_out in enumerate(self.F_h):
            if self.rename:
                linear_c = nn.Dense(f_out, use_bias=False, in_units=self.N_C, prefix='cond_{}'.format(i))
                linear_sol = nn.Dense(f_out, use_bias=False, in_units=5, prefix='l_sol_{}'.format(i))
            else:
                linear_c = nn.Dense(f_out, use_bias=False, in_units=self.N_C)
                linear_sol = nn.Dense(f_out, use_bias=False, in_units=5)
            self.register_child(linear_c)
            self.linear_c.append(linear_c)
            self.register_child(linear_sol)
            self.linear_sol.append(linear_sol)
        
        
    def _graph_conv_sol_forward(self,X_sol,A_sol,NX_sol):
        for conv in self.conv_sol:
            X_sol = conv([X_sol, A_sol])
        #print(X_sol,NX_sol)
        X_sol_avg = reduced_sum()([X_sol,NX_sol])
        #print(X_sol_avg)
        return self.activation(self.linear_sol_first(X_sol_avg))
    
    def _graph_conv_forward(self, X, A,NX,NX_rep, D_sol,c, ids):
        X_out = [X]
        X_pred_out = [X]
        for conv, bn, linear_c, linear_sol in zip(self.conv, self.bn, self.linear_c, self.linear_sol):
            X = X_out[-1]
            X_pred = X_pred_out[-1]
            if bn is not None:
                X_out.append(conv(self.activation(bn(X)), A) +  linear_c(c)[ids, :] + linear_sol(D_sol)[ids, :])
                X_pred_out.append(conv(self.activation(bn(X_pred)), A))
            else:
                X_out.append(conv(X, A) + linear_c(c)[ids, :] + linear_sol(D_sol)[ids, :] )
                X_pred_out.append(conv(X_pred, A))
        X_out = nd.concat(*X_out[1:], dim=1)
        return self.activation(self.linear_skip(self.activation(self.bn_skip(X_out))))
    
class GraphConvolution(gluon.Block):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    
    def __init__(self, units,input_dims,
                 activation='leakyrelu',
                 **kwargs):
        
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        if activation == 'leakyrelu':
            self.activation = nn.activations.LeakyReLU(alpha=0.1)
        else:
            self.activation = nn.Activation(activation)
        self.supports_masking = True
        self.input_dims = input_dims
        with self.name_scope():
            self.kernel = self.params.get('kernel',
                                           shape=(self.input_dims,self.units),
                                           init=mx.init.Xavier(magnitude=1),
                                           allow_deferred_init=True)
            self.bias = self.params.get('bias',
                                        shape=(self.units,),
                                        init=mx.init.Zero(),
                                        differentiable=False)
        self.kernel.initialize()
        self.bias.initialize()
        
    def forward(self, inputs):
        assert isinstance(inputs, list)
        features, basis = inputs
        supports = nd.dot(basis, features)
        output = nd.dot(supports, self.kernel.data()) #####
        if self.input_dims == self.units:
            output = output+features # Skip Connection 
        output = output + self.bias.data()
        result = self.activation(output)
        return output

class reduced_sum(nn.Block):
    
    def __init__(self, activation='relu',**kwargs):
        
        super(reduced_sum, self).__init__(**kwargs)
        
        #self.activation = nn.Activation(activation)

    def forward(self, inputs):
        assert isinstance(inputs, list)
        graph, NF = inputs
        graph_h = nd.dot(NF,graph)
        return graph_h


