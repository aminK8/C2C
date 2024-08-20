import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class CrossAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_seg_classes: int,
                 hidden_dim: int,
                 kernel_size: Tuple[int, int] = (1, 1),
                 seg_kernel_size: Tuple[int, int] = (1, 1),
                 num_heads: int = 8,
                 drop_p: float = 0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size

        self.query_conv = nn.Conv2d(num_seg_classes, hidden_dim, kernel_size=seg_kernel_size, stride=kernel_size)
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=kernel_size)
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=kernel_size)

        self.output_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.output_drop = nn.Dropout2d(drop_p)
        
    def forward(self, x, seg):
        batch_size, num_channels, h, w = x.size()

        assert h % self.kernel_size[0] == 0, "input h must be divisible by kernel h"
        assert w % self.kernel_size[1] == 0, "input w must be divisible by kernel w"
        out_h = h // self.kernel_size[0]
        out_w = w // self.kernel_size[1]

        query = self.query_conv(seg).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # (batch_size, num_head, seq_len, head_dim)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)  # (batch_size, num_head, head_dim, seq_len)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # (batch_size, num_head, seq_len, head_dim)

        attention_scores = (query @ key) * self.scale  # (batch_size, num_head, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_head, seq_len, seq_len)

        attention_output = attention_weights @ value  # (batch_size, num_head, seq_len, head_dim)
        attention_output = attention_output.permute(0, 1, 3, 2).reshape(batch_size, self.hidden_dim, -1)  # (batch_size, hidden_dim, out_h*out_w)
        attention_output = attention_output.view(batch_size, -1, out_h, out_w)  # (batch_size, hidden_dim, out_h, out_w)

        output = self.output_conv(attention_output)  # (batch_size, hidden_dim, out_h, out_w)
        output = self.output_drop(output)  # (batch_size, hidden_dim, out_h, out_w)

        return output
