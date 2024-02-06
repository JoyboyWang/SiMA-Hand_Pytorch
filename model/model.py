import logging
import torch
import torch.nn as nn
from pytorch3d import transforms as p3dt

from common.utils.mano import MANO

from model.mob_recon.utils.read import spiral_tramsform
from model.mob_recon.utils.utils import *
from model.mob_recon.models.modules import Upsample_MV

from model.mob_recon.models.transformer import *
from model.h2onet.conv.spiralconv import *
from model.h2onet.models.densestack import *
from model.h2onet.models.modules import *

from model.utils import *

logger = logging.getLogger(__name__)
mano = MANO()


class SVRHand(nn.Module):

    def __init__(self, cfg):
        super(SVRHand, self).__init__()
        self.cfg = cfg
        self.backbone = SIMA_Backnone(cfg=self.cfg, latent_size=256, kpts_num=21)
        template_fp = "model/h2onet/template/template.ply"
        transform_fp = "model/h2onet/template/transform.pkl"
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors=[2, 2, 2, 2], seq_length=[9, 9, 9, 9], dilation=[1, 1, 1, 1])
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())

        uv_channel=21
        num_vert = [u[0].size(0) // 3 for u in up_transform] + [up_transform[-1][0].size(0) // 6]
        self.upsample = nn.Parameter(torch.ones([num_vert[-1], uv_channel]) * 0.01, requires_grad=True) # torch.Size([49, 21])

        init_weights_flag = cfg.model.get("init_weights", False)
        self.decoder3d = Upsample_MV(out_channels=[32, 64, 128, 256],
                                        spiral_indices=spiral_indices,
                                        up_transform=up_transform,
                                        meshconv=SpiralConv,
                                        init_weights=init_weights_flag)

        self.rot_reg = SIMA_GlobRotReg()

        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)
        self.mano_joint_reg = torch.nn.Parameter(self.mano_joint_reg)

        self.de_layer_conv = conv_layer(256, 256, 1, bn=False, relu=False)

        self.fusion_skip_primary = self.cfg.data.get("fusion_skip_primary", True)

        self.shared_encoding_mapping = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.j_latent_mapping = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Hand Orientation Regression
        self.dense_stack3 = DenseStack2_Encoder(128)

    def index(self, feat, uv): # feat: torch.Size([B, C, 4, 4])
        # import ipdb; ipdb.set_trace()
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, input):
        xs = input["img"] # torch.Size([B, 3, 128, 128])
        batch_size, channel_num, image_width, image_height = xs.shape

        upsample = self.upsample.repeat(batch_size, 1, 1).to(xs.device)

        latent, j_latent, share_encoding, uv = self.backbone(xs.view(-1, channel_num, image_width, image_height)) # torch.Size([B, 256, 4, 4]) rot_latent&j_latent: torch.Size([BxV, 1024, 4, 4])

        share_encoding = share_encoding.detach()  # torch.Size([B, 128, 32, 32])
        share_encoding_residual = self.shared_encoding_mapping(share_encoding)  # torch.Size([B, 128, 32, 32])
        share_encoding_pooled_all = share_encoding + share_encoding_residual  # torch.Size([B, 128, 32, 32])
        rot_latent = self.dense_stack3(share_encoding_pooled_all)  # torch.Size([B, 1024, 4, 4])

        # get 2d joints
        pred2d_pt = uv.clone()

        # normalize the coordinates of joint
        uv = torch.clamp((uv - 0.5) * 2, -1, 1) # torch.Size([B, 21, 2])
        # generate image feature from latent feature
        latent_de_layer_conv = self.de_layer_conv(latent) # torch.Size([B, 256, 4, 4])
        # get feature from corresponding image
        x = self.index(latent_de_layer_conv, uv).permute(0, 2, 1) # torch.Size([B, 21, 256])
        # 3d hand mesh feature lifting
        encoding_all = torch.bmm(upsample, x) # torch.Size([B, 49, 256])

        # predict 3d vertex wo global rotation
        pred_verts_wo_gr = self.decoder3d(encoding_all) # torch.Size([B, 778, 3])
        # predict 3d joint wo global rotation
        pred_joints_wo_gr = torch.matmul(self.mano_joint_reg.to(pred_verts_wo_gr.device), pred_verts_wo_gr)  # torch.Size([B, 21, 3])

        # predict global rotation following H2ONet
        j_latent = j_latent.detach()  # torch.Size([B, 1024, 4, 4])
        # fuse j_latent and rot_latent
        j_latent_residual = self.j_latent_mapping(j_latent)  # torch.Size([B, 1024, 4, 4])
        # add residual to primary feature
        j_latent_pooled_all = j_latent + j_latent_residual

        pred_glob_rot = self.rot_reg(j_latent_pooled_all, rot_latent)
        pred_glob_rot_mat = p3dt.rotation_6d_to_matrix(pred_glob_rot)

        pred_root_joint_wo_gr = pred_joints_wo_gr[:, 0, None, ...]
        pred_verts_w_gr = torch.matmul(pred_verts_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1))
        pred_verts_w_gr = pred_verts_w_gr + pred_root_joint_wo_gr
        pred_joints_w_gr = torch.matmul(pred_joints_wo_gr.detach() - pred_root_joint_wo_gr.detach(), pred_glob_rot_mat.permute(0, 2, 1))
        pred_joints_w_gr = pred_joints_w_gr + pred_root_joint_wo_gr

        if "mano_pose" in input:
            gt_mano_params = torch.cat([input["mano_pose"], input["mano_shape"]], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts_w_gr, gt_joints_w_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
            gt_glob_rot = gt_mano_pose[:, :3].clone()
            gt_glob_rot_mat = p3dt.axis_angle_to_matrix(gt_glob_rot)
            gt_mano_pose[:, :3] = 0
            gt_verts_wo_gr, gt_joints_wo_gr = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts_w_gr /= 1000
            gt_joints_w_gr /= 1000
            gt_verts_wo_gr /= 1000
            gt_joints_wo_gr /= 1000

        else:
            gt_mano_params = None

        output = {}
        
        # for evaluation purpose
        output["pred_verts3d_cam"] = pred_verts_w_gr
        output["pred_joints3d_cam"] = pred_joints_w_gr
        
        output["pred_verts3d_wo_gr"] = pred_verts_wo_gr
        output["pred_joints3d_wo_gr"] = pred_joints_wo_gr
        output["pred_verts3d_w_gr"] = pred_verts_w_gr
        output["pred_joints3d_w_gr"] = pred_joints_w_gr
        output["pred_joints_img"] = pred2d_pt
        output["pred_glob_rot"] = pred_glob_rot
        output["pred_glob_rot_mat"] = pred_glob_rot_mat
        if gt_mano_params is not None:
            output["gt_verts3d_cam"] = gt_verts_w_gr
            output["gt_joints3d_cam"] = gt_joints_w_gr
            output["gt_glob_rot"] = gt_glob_rot
            output["gt_glob_rot_mat"] = gt_glob_rot_mat
            output["gt_verts3d_w_gr"] = gt_verts_w_gr
            output["gt_joints3d_w_gr"] = gt_joints_w_gr
            output["gt_verts3d_wo_gr"] = gt_verts_wo_gr
            output["gt_joints3d_wo_gr"] = gt_joints_wo_gr

            if "val_mano_pose" in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input["val_mano_pose"], th_betas=input["mano_shape"])
                output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"] = val_gt_verts / 1000, val_gt_joints / 1000
        
        else:
            output["gt_glob_rot_mat"] = input['gt_glob_rot_mat']
            output["gt_verts3d_w_gr"] = input['gt_verts3d_w_gr']
            output["gt_joints3d_w_gr"] = input['gt_joints3d_w_gr']
            output["gt_verts3d_wo_gr"] = input['gt_verts3d_wo_gr']
            output["gt_joints3d_wo_gr"] = input['gt_joints3d_wo_gr']

            if 'gt_glob_rot_mat_w_aug' in input:
                output['gt_glob_rot_mat_w_aug'] = input['gt_glob_rot_mat_w_aug']

        # for  computation
        output["joints_img"] = input["joints_img"]

        return output
