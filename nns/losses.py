import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class SSLLoss(nn.Module):
    """
    Modified from iBoT: https://github.com/bytedance/ibot/blob/main/main_ibot.py
    """

    def __init__(self,
                 embed_out_channels,
                 map_out_channels,
                 map_size,
                 num_views,
                 teacher_temp,
                 warmup_teacher_temp,
                 warmup_teacher_temp_epochs,
                 nepochs,
                 student_temp=0.1,
                 teacher_temp2=None,
                 warmup_teacher_temp2=None,
                 center_momentum_embed=0.9,
                 center_momentum_map=0.9,
                 lambda1=1.0,
                 lambda2=1.0,
                 world_size=1):
        super().__init__()
        self.num_views = num_views
        self.world_size = world_size

        self.student_temp = student_temp
        self.center_momentum_embed = center_momentum_embed
        self.center_momentum_map = center_momentum_map
        self.register_buffer("center_embed", torch.zeros(1, embed_out_channels))
        self.register_buffer("center_map", torch.zeros(1, map_out_channels, map_size[0], map_size[1]))

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        if teacher_temp2 is None:
            teacher_temp2, warmup_teacher_temp2 = teacher_temp, warmup_teacher_temp

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_embed, student_map = student_output
        teacher_embed, teacher_map = teacher_output
        num_samples = student_embed.shape[0] // self.num_views

        student_embed = student_embed / self.student_temp  # (batch_size * num_views, C)
        student_embed_chunk = student_embed.chunk(num_samples)  # (batch_size, num_views, C)
        student_map = student_map / self.student_temp  # (batch_size * num_views, C, H, W)
        student_map_chunk = student_map.chunk(num_samples)  # (batch_size, num_views, C, H, W)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_embed_chunk = F.softmax((teacher_embed - self.center_embed) / temp, dim=-1)  # (batch_size * num_views, C)
        teacher_embed_chunk = teacher_embed_chunk.detach().chunk(num_samples)
        teacher_map_chunk = F.softmax((teacher_map - self.center_map) / temp2, dim=1)  # (batch_size * num_views, C, H, W)
        teacher_map_chunk = teacher_map_chunk.detach().chunk(num_samples)

        student_mask_chunk = student_mask.detach().chunk(num_samples)  # (batch_size, num_views, H, W)

        assert len(teacher_embed_chunk) == len(student_embed_chunk)
        assert len(teacher_map_chunk) == len(student_map_chunk)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for si in range(len(teacher_embed_chunk)):
            for q in range(self.num_views):
                for v in range(self.num_views):
                    if v == q:
                        loss2 = torch.sum(-teacher_map_chunk[si][q] * F.log_softmax(student_map_chunk[si][v], dim=0), dim=0)  # (H, W)
                        # loss2 = torch.mean((teacher_map_chunk[si][q] - student_map_chunk[si][v])**2, dim=0)  # (H, W)
                        loss2 = torch.sum(loss2 * student_mask_chunk[si][v]) / student_mask_chunk[si][v].sum().clamp(min=1.0)
                        total_loss2 += loss2
                        n_loss_terms2 += 1
                    else:
                        loss1 = torch.sum(-teacher_embed_chunk[si][q] * F.log_softmax(student_embed_chunk[si][v], dim=-1), dim=-1)
                        total_loss1 += loss1
                        n_loss_terms1 += 1
            
        total_loss1 = (total_loss1 / n_loss_terms1) * self.lambda1
        total_loss2 = (total_loss2 / n_loss_terms2) * self.lambda2
        total_loss = dict(embed=total_loss1, map=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_embed, teacher_map)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_embed, teacher_map):
        """
        Update centers used for teacher output.
        """
        center_embed = torch.sum(teacher_embed, dim=0, keepdim=True)
        dist.all_reduce(center_embed)
        center_embed = center_embed / (len(teacher_embed) * self.world_size)
        self.center_embed = self.center_embed * self.center_momentum_embed + center_embed * (1 - self.center_momentum_embed)

        center_map = torch.sum(teacher_map, dim=0, keepdim=True)
        dist.all_reduce(center_map)
        center_map = center_map / (len(teacher_map) * self.world_size)
        self.center_map = self.center_map * self.center_momentum_map + center_map * (1 - self.center_momentum_map)
