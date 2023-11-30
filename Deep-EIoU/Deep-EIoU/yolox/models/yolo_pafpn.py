#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        # print("input", input.shape)
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        
        # print("x0", x0.shape)
        # print("x1", x1.shape)
        # print("x2", x2.shape)

        
        
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # print("\nfpn_out0", fpn_out0.shape)
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # print("upsampled fpn_out0", fpn_out0.shape)
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        # print("\nconcat with x1 f_out0", f_out0.shape)
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # print("after c3 f_out0", f_out0.shape)
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # print("after conv1 reduce fpn_out1", fpn_out1.shape)
        f_out1 = self.upsample(fpn_out1)  # 256/8


        # print("upsampled f_out1", f_out1.shape)

        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        # print("\nconcat with x2 f_out1", f_out1.shape)

        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # print("after c3 pan_out2", pan_out2.shape)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # print("after bu p_out1", p_out1.shape)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # print("\nconcat with p_out1 fpn_out1", p_out1.shape)
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # print("after c3 pan_out1", pan_out1.shape)

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # print("after bu p_out0", p_out0.shape)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # print("\nconcat with p_out0 fpn_out0", p_out0.shape)
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # print("after c3 pan_out0", pan_out0.shape)

        # print("\npan_out2", pan_out2.shape)
        # print("pan_out1", pan_out1.shape)
        # print("pan_out0", pan_out0.shape)
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
