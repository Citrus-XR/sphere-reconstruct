"""FastDVDnet の推論ネットワーク.

MIT licensed upstream implementation を PyTorch 2.x 向けに整理したもの。重みとの互換性を
保つため module 名と層構成は変更しない。

Reference:
https://github.com/m-tassano/fastdvdnet/blob/c8fdf6182a0340e89dd18f5df25b47337cbede6f/models.py
"""

from __future__ import annotations

import torch
from torch import nn


class CvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convblock(value)


class InputCvBlock(nn.Module):
    def __init__(self, num_in_frames: int, out_ch: int) -> None:
        super().__init__()
        intermediate_channels = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(
                num_in_frames * 4,
                num_in_frames * intermediate_channels,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False,
            ),
            nn.BatchNorm2d(num_in_frames * intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * intermediate_channels, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convblock(value)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convblock(value)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convblock(value)


class OutputCvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convblock(value)


class DenBlock(nn.Module):
    def __init__(self, num_input_frames: int = 3) -> None:
        super().__init__()
        self.inc = InputCvBlock(num_input_frames, 32)
        self.downc0 = DownBlock(32, 64)
        self.downc1 = DownBlock(64, 128)
        self.upc2 = UpBlock(128, 64)
        self.upc1 = UpBlock(64, 32)
        self.outc = OutputCvBlock(32, 3)

    def forward(
        self,
        in0: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        noise_map: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        return in1 - self.outc(x0 + x1)


class FastDVDnet(nn.Module):
    def __init__(self, num_input_frames: int = 5) -> None:
        super().__init__()
        if num_input_frames != 5:
            raise ValueError("FastDVDnet の入力は 5 frame 固定です")
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)

    def forward(self, value: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        frames = tuple(value[:, 3 * index : 3 * index + 3] for index in range(self.num_input_frames))
        stage0 = self.temp1(frames[0], frames[1], frames[2], noise_map)
        stage1 = self.temp1(frames[1], frames[2], frames[3], noise_map)
        stage2 = self.temp1(frames[2], frames[3], frames[4], noise_map)
        return self.temp2(stage0, stage1, stage2, noise_map)
