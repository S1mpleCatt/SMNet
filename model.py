from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def _output_length(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class TanhSigmoidDropout(nn.Module):
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.tanh(x) * torch.sigmoid(x))


class ResCUM(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 32,
        negative_slope: float = 0.2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 8, kernel_size=16, stride=1, padding=7)
        self.activation1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(8, feature_dim, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(input_dim, feature_dim, kernel_size=1)
        self.tsd = TanhSigmoidDropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.activation1(self.conv1(x))
        features = self.activation2(self.conv2(features))
        residual = self.conv3(x)
        length = min(features.size(-1), residual.size(-1))
        return self.tsd(features[..., :length] + residual[..., :length])


class SJFEBackbone(nn.Module):
    def __init__(
        self,
        joint_count: int = 6,
        input_dim: int = 3,
        feature_dim: int = 32,
        negative_slope: float = 0.2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.joint_count = joint_count
        self.branches = nn.ModuleList(
            [
                ResCUM(input_dim, feature_dim, negative_slope, dropout)
                for _ in range(joint_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joint_private_features = [
            branch(x[:, joint_index])
            for joint_index, branch in enumerate(self.branches)
        ]
        return torch.cat(joint_private_features, dim=1)


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: float = 0.5):
        super().__init__()
        hidden_channels = max(1, int(channels * reduction))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
        )
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average_descriptor = x.mean(dim=2)
        maximum_descriptor = x.amax(dim=2)
        channel_attention = torch.sigmoid(
            self.channel_mlp(average_descriptor)
            + self.channel_mlp(maximum_descriptor)
        ).unsqueeze(2)
        x = x * channel_attention
        average_map = x.mean(dim=1, keepdim=True)
        maximum_map = x.amax(dim=1, keepdim=True)
        temporal_attention = torch.sigmoid(
            self.temporal_conv(torch.cat((average_map, maximum_map), dim=1))
        )
        return x * temporal_attention


class ContextualSaliencyAttention(nn.Module):
    def __init__(self, channels: int, reduction: float = 0.5):
        super().__init__()
        hidden_channels = max(1, int(channels * reduction))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
        )
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average_descriptor = x.mean(dim=2)
        maximum_descriptor = x.amax(dim=2)
        channel_attention = torch.sigmoid(
            self.channel_mlp(average_descriptor + maximum_descriptor)
        ).unsqueeze(2)
        x = x * channel_attention
        average_map = x.mean(dim=1, keepdim=True)
        maximum_map = x.amax(dim=1, keepdim=True)
        temporal_attention = torch.sigmoid(
            self.temporal_conv(torch.cat((average_map, maximum_map), dim=1))
        )
        return x * temporal_attention


class DualPathAttention(nn.Module):
    def __init__(self, channels: int, reduction: float = 0.5):
        super().__init__()
        self.cbam1d_path = CBAM1D(channels, reduction)
        self.csa_path = ContextualSaliencyAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.cbam1d_path(x) + self.csa_path(x))


class CascadedDilatedConvolutionBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 16,
        dilation: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.initial_conv = nn.Conv1d(
            input_channels,
            16,
            kernel_size=64,
            stride=8,
            padding=31,
        )
        self.activation = nn.LeakyReLU(negative_slope)
        self.initial_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dilated_conv = nn.Conv1d(
            16,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.tsd = TanhSigmoidDropout(dropout)
        self.output_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_pool(self.activation(self.initial_conv(x)))
        x = self.tsd(self.dilated_conv(x))
        return self.output_pool(x)


class AGDFNeck(nn.Module):
    def __init__(
        self,
        input_channels: int = 192,
        branch_channels: int = 16,
        dilation_rates: Sequence[int] = (1, 2, 3),
        attention_reduction: float = 0.5,
        negative_slope: float = 0.2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.pre_attention = DualPathAttention(
            input_channels,
            attention_reduction,
        )
        self.cdcbs = nn.ModuleList(
            [
                CascadedDilatedConvolutionBlock(
                    input_channels,
                    branch_channels,
                    dilation,
                    negative_slope,
                    dropout,
                )
                for dilation in dilation_rates
            ]
        )
        output_channels = branch_channels * len(dilation_rates)
        self.post_attention = DualPathAttention(
            output_channels,
            attention_reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_attention(x)
        multi_scale_features = [cdcb(x) for cdcb in self.cdcbs]
        return self.post_attention(torch.cat(multi_scale_features, dim=1))


class MambaLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 48,
        state_dim: int = 48,
        convolution_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is required to construct SMNet. Install mamba-ssm before creating the model."
            )
        self.mamba = Mamba(
            d_model=embedding_dim,
            d_state=state_dim,
            d_conv=convolution_width,
            expand=expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mamba(x))


class MambaMixer(nn.Module):
    def __init__(
        self,
        input_dim: int = 48,
        embedding_dim: int = 48,
        layer_count: int = 3,
        state_dim: int = 48,
        convolution_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_projection = (
            nn.Identity()
            if input_dim == embedding_dim
            else nn.Linear(input_dim, embedding_dim)
        )
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    embedding_dim,
                    state_dim,
                    convolution_width,
                    expansion,
                    dropout,
                )
                for _ in range(layer_count)
            ]
        )
        self.output_projection = (
            nn.Identity()
            if input_dim == embedding_dim
            else nn.Linear(embedding_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_projection(x)
        return x.transpose(1, 2)


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        self.activation = nn.Tanh()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.batch_norm(self.conv(self.activation(x))))


class Head(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_length: int,
        class_count: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvolutionBlock(input_channels, 64),
            ConvolutionBlock(64, 32),
            ConvolutionBlock(32, 32),
        )
        output_length = input_length
        for _ in range(3):
            output_length = _output_length(output_length, kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * output_length, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_blocks(x))


class SMNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        joint_count: int = 6,
        signal_length: int = 2560,
        class_count: int = 6,
        feature_dim: int = 32,
        branch_channels: int = 16,
        dilation_rates: Sequence[int] = (1, 2, 3),
        mamba_embedding_dim: int = 48,
        mamba_layers: int = 3,
        mamba_state_dim: int = 48,
        mamba_convolution_width: int = 4,
        mamba_expansion: int = 2,
        mamba_dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.joint_count = joint_count
        self.signal_length = signal_length
        self.sjfe_backbone = SJFEBackbone(
            joint_count=joint_count,
            input_dim=input_dim,
            feature_dim=feature_dim,
        )
        sjfe_channels = joint_count * feature_dim
        self.agdf_neck = AGDFNeck(
            input_channels=sjfe_channels,
            branch_channels=branch_channels,
            dilation_rates=dilation_rates,
        )
        agdf_channels = branch_channels * len(dilation_rates)
        self.mamba_mixer = MambaMixer(
            input_dim=agdf_channels,
            embedding_dim=mamba_embedding_dim,
            layer_count=mamba_layers,
            state_dim=mamba_state_dim,
            convolution_width=mamba_convolution_width,
            expansion=mamba_expansion,
            dropout=mamba_dropout,
        )
        sjfe_length = _output_length(signal_length, kernel_size=16, padding=7)
        agdf_length = _output_length(
            sjfe_length,
            kernel_size=64,
            stride=8,
            padding=31,
        )
        agdf_length = _output_length(agdf_length, kernel_size=2, stride=2)
        agdf_length = _output_length(agdf_length, kernel_size=2, stride=2)
        self.head = Head(
            input_channels=agdf_channels,
            input_length=agdf_length,
            class_count=class_count,
        )

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        expected_channels = self.joint_count * self.input_dim
        if x.ndim == 3 and x.size(1) == expected_channels:
            x = x.reshape(
                x.size(0),
                self.joint_count,
                self.input_dim,
                x.size(2),
            )
        elif x.ndim == 3 and x.size(2) == expected_channels:
            x = x.transpose(1, 2).contiguous().reshape(
                x.size(0),
                self.joint_count,
                self.input_dim,
                x.size(2),
            )
        if x.ndim != 4:
            raise ValueError(
                "SMNet expects [B, J, M, T], [B, J*M, T], or [B, T, J*M]."
            )
        if x.size(1) != self.joint_count or x.size(2) != self.input_dim:
            raise ValueError(
                f"Expected {self.joint_count} joints and {self.input_dim} axes per joint, got {x.size(1)} and {x.size(2)}."
            )
        if x.size(3) != self.signal_length:
            raise ValueError(
                f"Expected signal length {self.signal_length}, got {x.size(3)}."
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        joint_private_features = self.sjfe_backbone(x)
        fused_features = self.agdf_neck(joint_private_features)
        mixed_features = self.mamba_mixer(fused_features)
        return self.head(mixed_features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
