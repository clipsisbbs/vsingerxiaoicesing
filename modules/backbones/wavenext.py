import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams


class Conv1d(torch.nn.Conv1d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		nn.init.kaiming_normal_(self.weight)

class qkrelu(nn.Module):
	def __init__(self):
		super().__init__()
		self.relu = nn.LeakyReLU(negative_slope=0.2)
		self.offset = nn.Parameter(torch.tensor(0.0),requires_grad=True)
	def forward(self, x):
		return self.relu(x + self.offset) - self.offset

class ResidualBlock(nn.Module):
	def __init__(self, encoder_hidden, residual_channels, expansion, dilation_index=0):
		super().__init__()
		self.residual_channels = residual_channels
		self.relu = qkrelu()
		self.dilated_conv = nn.Sequential(
			nn.Conv1d(residual_channels, 2 * residual_channels, 1),
			nn.SiLU(),
			nn.Conv1d(
				2 * residual_channels,
				2 * residual_channels,
				kernel_size=expansion * 2 + 1,
				padding="same",
				dilation=(expansion + 1) ** dilation_index,
				groups=residual_channels
			)
		)
		self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
		self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
		self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
		self.gate_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
	def forward(self, x, conditioner, diffusion_step):
		diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
		conditioner = self.conditioner_projection(conditioner)
		y = x + diffusion_step

		y = self.dilated_conv(y) + conditioner

		# Using torch.split instead of torch.chunk to avoid using onnx::Slice
		gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
		y = torch.sigmoid(gate) * self.relu(filter)

		y = self.output_projection(y)

		# Using torch.split instead of torch.chunk to avoid using onnx::Slice
		residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
		return (x + residual) / math.sqrt(2.0), skip


class WaveNeXt(nn.Module):
	def __init__(self, in_dims, n_feats, *, num_layers=20, num_channels=256, kernel_size=7, dilation_cycle_length=3):
		super().__init__()
		self.in_dims = in_dims
		self.n_feats = n_feats
		self.input_projection = Conv1d(in_dims * n_feats, num_channels, 1)
		self.diffusion_embedding = SinusoidalPosEmb(num_channels)
		self.mlp = nn.Sequential(
			nn.Linear(num_channels, num_channels * 4),
			nn.Mish(),
			nn.Linear(num_channels * 4, num_channels)
		)
		self.residual_layers = nn.ModuleList([
			ResidualBlock(
				encoder_hidden=hparams['hidden_size'],
				residual_channels=num_channels,
				expansion=(kernel_size - 1) // 2,
				dilation_index=i % dilation_cycle_length
			)
			for i in range(num_layers)
		])
		self.skip_projection = Conv1d(num_channels, num_channels, 1)
		self.output_projection = Conv1d(num_channels, in_dims * n_feats, 1)
		nn.init.zeros_(self.output_projection.weight)

	def forward(self, spec, diffusion_step, cond):
		"""
		:param spec: [B, F, M, T]
		:param diffusion_step: [B, 1]
		:param cond: [B, H, T]
		:return:
		"""
		if self.n_feats == 1:
			x = spec.squeeze(1)  # [B, M, T]
		else:
			x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
		x = self.input_projection(x)  # [B, C, T]

		x = F.relu(x)
		diffusion_step = self.diffusion_embedding(diffusion_step)
		diffusion_step = self.mlp(diffusion_step)
		skip = []
		for layer in self.residual_layers:
			x, skip_connection = layer(x, cond, diffusion_step)
			skip.append(skip_connection)

		x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
		x = self.skip_projection(x)
		x = F.relu(x)
		x = self.output_projection(x)  # [B, M, T]
		if self.n_feats == 1:
			x = x[:, None, :, :]
		else:
			# This is the temporary solution since PyTorch 1.13
			# does not support exporting aten::unflatten to ONNX
			# x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
			x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
		return x