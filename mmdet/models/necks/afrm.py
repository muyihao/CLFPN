import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d
from collections import OrderedDict
from ..builder import NECKS
from mmcv.runner import BaseModule


@NECKS.register_module()
class AFRM(BaseModule):
    """

  

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.

            [None, 'conv', 'non_local'].
    """



    def __init__(self,
                 in_channels,
                 num_levels,
                 k_size = 3,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(AFRM, self).__init__()


        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level

        assert 0 <= self.refine_level < self.num_levels


        # level attention
        self.level_conv = nn.Conv2d(self.num_levels, self.num_levels, 1)
        self.level_relu = nn.ReLU(inplace=True)
        self.level_sigmoid = nn.Hardsigmoid()
        
        # spatial module
        self.spatial_module = NonLocal2d(
            self.in_channels,
            reduction=1,
            use_scale=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        # channel module
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_max = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.cag_sigmoid = nn.Sigmoid()







    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)


        # step 2: refine gathered features
        # level attention
        # avg pool -> conv 1x1 relu hard sigmoid
        level_feature = torch.stack(feats, dim=1)  # N,L,C,H,W
        N, L, C, H, W = level_feature.size()
        level_feature = level_feature.reshape(N, L, C, -1).permute(0, 1, 3, 2) # N,L,S,C

        mid_feature = F.adaptive_max_pool2d(level_feature, 1)  # N,L,1,1
        mid_feature = self.level_conv(mid_feature)
        mid_feature = self.level_relu(mid_feature)# N,1,1 ,L
        level_att = self.level_sigmoid(mid_feature)# N, L, 1,1
        level_feature = level_feature * level_att # N,L,S,C
        #  gathered features
        # level_feature = torch.mean(level_feature, dim=1).view(level_feature.size()[0], -1, gather_size[0], gather_size[1])   # N,C,H,W
        level_feature = torch.mean(level_feature, dim=1).squeeze(1).permute(0, 2, 1).reshape(N, C, H, W)





        # spatial attention
        bsf = self.spatial_module(level_feature)

        # CAG
        brach1 = F.adaptive_avg_pool2d(bsf, 1)  # N,c,1,1
        brach1 = self.conv_avg(brach1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        brach2 = F.adaptive_max_pool2d(bsf, 1)  # N,c,1,1
        brach2 = self.conv_max(brach2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        channel_weight = self.cag_sigmoid(brach1 + brach2)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            # shortcut connection
            outs.append(residual + inputs[i]+channel_weight)

        return tuple(outs)


