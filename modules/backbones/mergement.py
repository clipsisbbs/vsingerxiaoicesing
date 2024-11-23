import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams

from modules.backbones.wavenet import ResidualBlock
from modules.backbones.lynxnet import LYNXNetResidualLayer

class PackedBlock(nn.Module):
	def __init__(self, dim=512, dim_cond=256, wn_seq=[0, 1, 2, 3], dilation_cycle_length=4):
		super().__init__()
		self.dim = dim
		self.wavenet = nn.ModuleList([
			ResidualBlock(
				encoder_hidden=dim_cond,
				residual_channels=dim,
				dilation=2 ** (i % dilation_cycle_length)
			)
			for i in wn_seq
		])
		self.lynxnet1 = LYNXNetResidualLayer(dim_cond, dim * 2, 2, dropout=0.1)
		self.lynxnet2 = LYNXNetResidualLayer(dim_cond, dim * 2, 2, dropout=0.1)
		self.wn_to_lynx = nn.Conv1d(dim, dim * 2, 1)
		self.wn_norm = nn.LayerNorm(dim)
		self.lynx_to_wn = nn.Sequential(
			nn.Conv1d(dim * 2, dim * 2, 1),
			nn.SiLU(),
			nn.Conv1d(dim * 2, dim, 1)
		)
		self.lynx_step = nn.Conv1d(dim, dim * 2, 1)
		self.norm = nn.LayerNorm(dim * 3)
	def forward(self, x, cond, diffstep):
		# print(diffstep.shape)
		wn_x, lynx_x = torch.split(x, [self.dim, self.dim * 2], dim=-2)
		connects = []
		for layer in self.wavenet:
			wn_x, conn_v = layer(wn_x, cond, diffstep)
			connects.append(conn_v)
		connects = torch.sum(torch.stack(connects), dim=0) / sqrt(len(connects))
		lynx_x = self.lynxnet1(lynx_x, cond, self.lynx_step(diffstep.unsqueeze(-1)))
		lynx_x = self.lynxnet2(lynx_x, cond, self.lynx_step(diffstep.unsqueeze(-1)))
		connects = self.wn_norm(connects.transpose(1,2)).transpose(1,2)
		x = torch.cat([wn_x + self.lynx_to_wn(lynx_x), lynx_x + self.wn_to_lynx(connects)], dim=-2)
		return self.norm(x.transpose(1,2)).transpose(1,2)

class Mergement(nn.Module):
	def __init__(self, in_dims, n_feats, *, num_channels=512, **kwargs):
		super().__init__()
		self.in_dims = in_dims
		self.n_feats = n_feats
		self.input_projection = nn.Conv1d(in_dims * n_feats, num_channels * 3, 1)
		self.diffusion_embedding = nn.Sequential(
			SinusoidalPosEmb(num_channels),
			nn.Linear(num_channels, num_channels * 4),
			nn.GELU(),
			nn.Linear(num_channels * 4, num_channels),
		)
		self.layers = nn.ModuleList([
			PackedBlock(num_channels, hparams['hidden_size'], [0, 1, 2, 3]),
			PackedBlock(num_channels, hparams['hidden_size'], [4, 5, 6]),
			PackedBlock(num_channels, hparams['hidden_size'], [7, 8, 9, 10]),
			PackedBlock(num_channels, hparams['hidden_size'], [11, 12, 13]),
			PackedBlock(num_channels, hparams['hidden_size'], [14, 15, 16, 17]),
			PackedBlock(num_channels, hparams['hidden_size'], [18, 19]),
		])
		self.output_projection = nn.Conv1d(num_channels * 3, in_dims * n_feats, kernel_size=1)
	def forward(self, spec, diffusion_step, cond):
		"""
		:param spec: [B, F, M, T]
		:param diffusion_step: [B, 1]
		:param cond: [B, H, T]
		:return:
		"""
		
		if self.n_feats == 1:
			x = spec[:, 0]  # [B, M, T]
		else:
			x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
	
		x = self.input_projection(x)  # x [B, residual_channel, T]
		diffusion_step = self.diffusion_embedding(diffusion_step)
		for layer in self.layers:
			x = layer(x, cond, diffusion_step)
		# MLP
		x = self.output_projection(x)  # [B, 128, T]
		
		if self.n_feats == 1:
			x = x[:, None, :, :]
		else:
			# This is the temporary solution since PyTorch 1.13
			# does not support exporting aten::unflatten to ONNX
			# x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
			x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
		return x
