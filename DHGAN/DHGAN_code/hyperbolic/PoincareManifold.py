#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
from typing import Any, Union
from hyperbolic.utils import *
import math

class PoincareManifold:

    def __init__(self, args, logger, EPS=1e-5, PROJ_EPS=1e-5):
        self.args = args
        self.logger = logger
        self.EPS = EPS
        self.min_norm = 1e-5
        self.PROJ_EPS = PROJ_EPS
        self.tanh = F.tanh
        self.cur_k = 1 # cur_k = -1/curvature, cur_k = 3, 2, 1, 1/2, 1/3, curvature= -1/3, -1/2, -1, -2, -3


    def normalize(self, x):
        return clip_by_norm(x, (1.0/self.cur_k - self.PROJ_EPS))

    def init_embed(self, embed, irange=1e-2):
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def mob_add(self, u, v):
        """
        Add two vectors in hyperbolic space
        """
        v = v + self.EPS
        th_dot_u_v = 2. * self.cur_k * th_dot(u, v)
        th_norm_u_sq = self.cur_k * th_dot(u, u)
        th_norm_v_sq = self.cur_k * th_dot(v, v)
        denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
        result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + self.EPS) * u + \
                 (1. - th_norm_u_sq) / (denominator + self.EPS) * v
        return self.normalize(result)

    def mob_scalar_multi(self, v, c):
        v_norm = th_norm(v)
        v_norm = torch.clamp(v_norm, min=self.min_norm)
        result = self.tanh(c * th_atanh(math.sqrt(self.cur_k) * v_norm)) / (v_norm * math.sqrt(self.cur_k)) * v
        return self.normalize(result)

    def mob_matrix_multi(self, v, m):
        v_norm = th_norm(v)
        v_norm = torch.clamp(v_norm, min=self.min_norm)
        mv = v @ m.transpose(-1, -2) + self.EPS
        mv_norm = th_norm(mv)
        result = self.tanh(mv_norm/v_norm * th_atanh(math.sqrt(self.cur_k) * v_norm)) / (mv_norm * math.sqrt(self.cur_k)) * mv
        return self.normalize(result)

    def distance(self, u, v):
        uv = self.mob_add(-u, v)
        uv_norm = th_norm(uv)
        uv_norm = torch.clamp(uv_norm, min=self.min_norm)
        dist = 2 * th_atanh(math.sqrt(self.cur_k) * uv_norm) / math.sqrt(self.cur_k)
        return dist

    def lambda_x(self, x):
        """
        A conformal factor
        """
        return 2. / (1 - self.cur_k * th_dot(x, x))

    def log_map_zero(self, y):
        norm_diff = th_norm(y)
        norm_diff = torch.clamp(norm_diff, min=self.min_norm)
        return 1. / th_atanh(math.sqrt(self.cur_k) * norm_diff, self.EPS) / (math.sqrt(self.cur_k) * norm_diff) * y

    def log_map_x(self, x, y):
        diff = self.mob_add(-x, y) + self.EPS
        norm_diff = th_norm(diff)
        lam = self.lambda_x(x)
        return (( 2. / (math.sqrt(self.cur_k) * lam)) * th_atanh(math.sqrt(self.cur_k) * norm_diff, self.EPS) / norm_diff) * diff

    def metric_tensor(self, x, u, v):
        """
        The metric tensor in hyperbolic space.
        In-place operations for saving memory. (do not use this function in forward calls)
        """
        u_dot_v = th_dot(u, v)
        lambda_x = self.lambda_x(x)
        lambda_x *= lambda_x
        lambda_x *= u_dot_v
        return lambda_x

    def exp_map_zero(self, v):
        """
        Exp map from tangent space of zero to hyperbolic space
        Args:
            v: [batch_size, *] in tangent space
        """
        norm_v = th_norm(v) # [batch_size, 1]
        norm_v = torch.clamp(norm_v, min=self.min_norm)
        result = self.tanh(math.sqrt(self.cur_k) * norm_v) / (math.sqrt(self.cur_k) * norm_v) * v
        return self.normalize(result)

    def exp_map_x(self, x, v):
        """
        Exp map from tangent space of x to hyperbolic space
        """
        norm_v = th_norm(v)  # [batch_size, 1]
        norm_v = torch.clamp(norm_v, min=self.min_norm)
        second_term = (self.tanh(math.sqrt(self.cur_k) * self.lambda_x(x) * norm_v / 2) / (math.sqrt(self.cur_k) * norm_v)) * v
        return self.normalize(self.mob_add(x, second_term))

    def gyr(self, u, v, w):  # not used
        u_norm = th_dot(u, u)
        v_norm = th_dot(v, v)
        u_dot_w = th_dot(u, w)
        v_dot_w = th_dot(v, w)
        u_dot_v = th_dot(u, v)
        A = - u_dot_w * v_norm + v_dot_w + 2 * u_dot_v * v_dot_w
        B = - v_dot_w * u_norm - u_dot_w
        D = 1 + 2 * u_dot_v + u_norm * v_norm
        return w + 2 * (A * u + B * v) / (D + self.EPS)

    def parallel_transport(self, src, dst, v):  # not used
        return self.lambda_x(src) / th.clamp(self.lambda_x(dst), min=self.EPS) * self.gyr(dst, -src, v)

    def rgrad(self, p, d_p):
        """
        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincare ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4.0).expand_as(d_p)
        return d_p

