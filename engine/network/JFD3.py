"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register



@register()
class JFD3(nn.Module):
    __inject__ = ["deblur",'backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        deblur: nn.Module,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        guide = False,
        brloss =False,
    ):
        super().__init__()
        self.deblur = deblur
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.guide=guide
        self.brloss = brloss
    def forward(self, x, targets=None,file_name=None):
        (blur_img,clear_img)=x
        blur_inner_feats, clear_inner_feats, deblurred_img = self.deblur(blur_img,clear_img)
       
      
        if self.guide:
                       
                x_backbone = self.backbone(deblurred_img,blur_inner_feats[3],file_name=file_name) 
                if self.brloss:
                    clear_x_backbone = self.backbone(clear_inner_feats[-1],blur_inner_feats[3],file_name=file_name)
                   
        else:
            x_backbone = self.backbone(deblurred_img,file_name=file_name) 
            if self.brloss:
                clear_x_backbone = self.backbone(clear_inner_feats[-1],file_name=file_name)

        x = self.encoder(x_backbone)
        x = self.decoder(x, targets)

        if self.brloss:
            x.update({
                'blur_inner_feats': blur_inner_feats,
                'clear_inner_feats': clear_inner_feats,
                'blur_backbone_feats': x_backbone,
                'clear_backbone_feats': clear_x_backbone,
            })
        else:
            x.update({
                'blur_inner_feats': blur_inner_feats,
                'clear_inner_feats': clear_inner_feats,
                })
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
   
        """
        保存特征图的通道平均图像（彩色，统一resize为640x640）
        :param feat: torch.Tensor, shape [B, C, H, W]
        :param save_path_suffix: str, 保存文件名后缀
        :param save_dir: str, 保存目录
        """

        import os
        import torch
        import cv2 
        import numpy as np
        file_name = file_name.split(".")[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        for i in range(feat.shape[0]):  # batch
            fmap = feat[i].mean(axis=0)  # [H, W]
            fmap -= fmap.min()
            fmap /= (fmap.max() + 1e-5)
            fmap = (fmap * 255).astype(np.uint8)
            # 变成彩色
            fmap_color = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
            # resize到640x640
            fmap_color = cv2.resize(fmap_color, (640, 640), interpolation=cv2.INTER_LINEAR)
            save_path = os.path.join(save_dir, f'{file_name}_{save_path_suffix}.png')
            cv2.imwrite(save_path, fmap_color)
        """
        保存三通道特征图为正常的RGB图片或灰度图（统一resize为640x640）
        :param feat: torch.Tensor, shape [B, 3, H, W] 或 [B, C, H, W] (C>=3时取前3个通道)
        :param save_path_suffix: str, 保存文件名后缀
        :param file_name: str, 原始文件名
        :param save_dir: str, 保存目录
        :param save_as_gray: bool, 是否保存为灰度图，True为灰度图，False为RGB图
        """
        file_name = file_name.split(".")[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        save_as_gray =False
        for i in range(feat.shape[0]):  # batch
            if save_as_gray:
                # 保存为灰度图：取通道平均值
                gray_feat = feat[i].mean(axis=0)  # [H, W]
                gray_feat -= gray_feat.min()
                gray_feat /= (gray_feat.max() + 1e-5)
                gray_feat = (gray_feat * 255).astype(np.uint8)
                
                # resize到640x640
                gray_feat = cv2.resize(gray_feat, (640, 640), interpolation=cv2.INTER_LINEAR)
                
                save_path = os.path.join(save_dir, f'{file_name}.jpg')
                cv2.imwrite(save_path, gray_feat)
            else:
                # 保存为RGB图
                # 取前3个通道作为RGB
                if feat.shape[1] >= 3:
                    rgb_feat = feat[i, :3]  # [3, H, W]
                else:
                    # 如果通道数少于3，则重复最后一个通道
                    rgb_feat = np.repeat(feat[i:i+1], 3, axis=0)  # [3, H, W]
                
                # 转换为HWC格式
                rgb_feat = np.transpose(rgb_feat, (1, 2, 0))  # [H, W, 3]
                
                # 对每个通道分别归一化，避免颜色偏差
                for c in range(3):
                    rgb_feat[:, :, c] -= rgb_feat[:, :, c].min()
                    rgb_feat[:, :, c] /= (rgb_feat[:, :, c].max() + 1e-5)
                
                rgb_feat = (rgb_feat * 255).astype(np.uint8)
                
                # resize到640x640
                rgb_feat = cv2.resize(rgb_feat, (640, 640), interpolation=cv2.INTER_LINEAR)
                
                save_path = os.path.join(save_dir, f'{file_name}_{save_path_suffix}_rgb.png')
                cv2.imwrite(save_path, rgb_feat)