import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import json
import math
import copy
from common.utils.preprocessing import load_img, process_bbox, augmentation, get_bbox
from common.utils.transforms import align_w_scale
from common.utils.mano import MANO

mano = MANO()


class HanCo(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split if data_split == "train" else "test"
        self.root_dir = cfg.data.root_dir
        self.annot_path = osp.join(self.root_dir, "annotations")
        self.root_joint_idx = 0

        self.datalist = self.load_data()

    def load_data(self):
        use_has_fit = self.cfg.data.get('use_has_fit', True)
        data_step_size = self.cfg.data.get('step_size', 1)
        aug_lst = self.cfg.data.get('read_aug', ['rgb'])
        aug_str = '-'.join(aug_lst)
        no_bbox_crop = self.cfg.data.get('no_bbox_crop', False)

        if use_has_fit:  # by default use dataset generated with has_fit
            if data_step_size == 1:
                json_path = osp.join(self.annot_path, f"HanCo_{self.data_split}_mv_data_{aug_str}-has_fit.json")
            else:
                json_path = osp.join(self.annot_path, f"HanCo_{self.data_split}_mv_data_{aug_str}-has_fit_{data_step_size}.json")
        else:  # else is_valid
            json_path = osp.join(self.annot_path, f"HanCo_{self.data_split}_mv_data_{aug_str}-is_valid_{data_step_size}_occ_ratio.json")


        print(json_path)

        f = open(json_path, 'r')
        db = json.load(f)

        datalist = []
        for idx, ann in enumerate(db['annotations']):  # for each hand scenarios
            # assert len(ann) == 8, "Not eight views!"
            for view_idx, view_ann in enumerate(ann):  # for each view
                # image_id = view_ann["image_id"]

                img = db['images'][idx][view_idx]
                img_path = osp.join(self.root_dir, img["file_name"])
                img_shape = (img["height"], img["width"])

                if self.data_split == "train":
                    joints_coord_cam = np.array(view_ann["joints_coord_cam"], dtype=np.float32)  # meter
                    cam_param = {k: np.array(v, dtype=np.float32) for k, v in view_ann["cam_param"].items()}
                    joints_coord_img = np.array(view_ann["joints_img"], dtype=np.float32)

                    bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=self.cfg.data.get("bbox_expansion_factor", 1.5))  # arbitrary size
                    # bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)  # arbitrary size
                    bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)  # width = height

                    if bbox is None:  # it discards something...
                        continue

                    if no_bbox_crop:
                        # if the bbox exists, we use the whole image as input
                        bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

                    # here, pose is different from the original one becase we did some original processing on the input to generate xxx_data.json
                    mano_pose = np.array(view_ann["mano_param"]["pose"], dtype=np.float32)  # length: 48
                    mano_shape = np.array(view_ann["mano_param"]["shape"], dtype=np.float32)
                    mano_trans = np.array(view_ann["mano_param"]["trans"], dtype=np.float32)

                    view_data = {
                        "img_path": img_path,
                        "img_shape": img_shape,
                        "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "bbox": bbox,
                        "cam_param": cam_param,
                        "mano_pose": mano_pose,
                        "mano_shape": mano_shape,
                        "mano_trans": mano_trans,
                    }
                else:
                    joints_coord_cam = np.array(view_ann["joints_coord_cam"], dtype=np.float32)
                    root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                    joints_coord_img = np.array(view_ann["joints_img"], dtype=np.float32)

                    bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=self.cfg.data.get("bbox_expansion_factor", 1.5))
                    # bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                    bbox = process_bbox(bbox, img["width"], img["height"], aspect_ratio=1, expansion_factor=1.0)
                    if bbox is None:
                        bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

                    if no_bbox_crop:
                        # if the bbox exists, we use the whole image as input
                        bbox = np.array([0, 0, img["width"] - 1, img["height"] - 1], dtype=np.float32)

                    cam_param = {k: np.array(v, dtype=np.float32) for k, v in view_ann["cam_param"].items()}

                    mano_pose = np.array(view_ann["mano_param"]["pose"], dtype=np.float32)
                    mano_shape = np.array(view_ann["mano_param"]["shape"], dtype=np.float32)
                    mano_trans = np.array(view_ann["mano_param"]["trans"], dtype=np.float32)

                    view_data = {
                        "img_path": img_path,
                        "img_shape": img_shape,
                        "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "root_joint_cam": root_joint_cam,
                        "bbox": bbox,
                        "cam_param": cam_param,
                        "mano_pose": mano_pose,
                        "mano_shape": mano_shape,
                        "mano_trans": mano_trans,
                    }
                datalist.append(view_data)

        print(len(datalist))  # train_full: 50861, test_full: 9846
        return datalist

    def __len__(self):
        return len(self.datalist)  # datalist detemines the number of iteratons

    def __getitem__(self, idx):
        data_item = copy.deepcopy(self.datalist[idx])

        data_augment = True
        #### common attributes ####
        # images
        img_path, img_shape, bbox, cam_param = data_item["img_path"], data_item["img_shape"], data_item["bbox"], data_item["cam_param"]
        img = load_img(img_path)  # (480, 640, 3）

        if data_augment:
            img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.cfg.data.input_img_shape, do_flip=False)  # img_color

        img = self.transform(img.astype(np.float32)) / 255.  # normalization # torch.Size([3, 128, 128])

        # mano parameter
        mano_pose, mano_shape, mano_trans = data_item["mano_pose"], data_item["mano_shape"], data_item["mano_trans"]

        # 3D joint camera coordinate
        joints_coord_cam = data_item["joints_coord_cam"]

        # 2D joint coordinate
        joints_img = data_item["joints_coord_img"]

        if self.data_split == "train":

            if data_augment:
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]  # [128, 128]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]  # [128, 128]

            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative

            # 3D data rotation augmentation
            if data_augment:
                rot_aug_mat = np.array(
                    [[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
                joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

                # 3D data rotation augmentation
                mano_pose = mano_pose.reshape(-1, 3)

                root_pose = mano_pose[self.root_joint_idx, :]  # 1x3, rotation vector
                root_pose, _ = cv2.Rodrigues(root_pose)  # 3x3, rotation matrix
                root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))  # 3x1, rotation vector
                mano_pose[self.root_joint_idx] = root_pose.reshape(3)
                mano_pose = mano_pose.reshape(-1)

        else:
            root_joint_cam = data_item["root_joint_cam"]

            # Only for rotation metric
            val_mano_pose = copy.deepcopy(mano_pose).reshape(-1, 3)
            val_mano_pose = val_mano_pose.reshape(-1)

            if data_augment:
                joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
                joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

        if self.data_split == "train":
            input = {
                "img": img,  # (8, 3, 128, 128)
                "bbox": bbox,  # (8, 4)
                "joints_img": joints_img,  # (8, 21, 2)
                "joints_coord_cam": joints_coord_cam,  # (8, 21, 3)
                "mano_pose": mano_pose,  # (8, 48)
                "mano_shape": mano_shape,  # (8, 10)
                "mano_trans": mano_trans,
                "root_joint_cam": root_joint_cam,  # (8, 3)
                "do_flip": False
            }
        else:

            input = {
                "img": img,
                "bbox": bbox,
                "joints_img": joints_img,
                "joints_coord_cam": joints_coord_cam,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
                "mano_trans": mano_trans,
                "val_mano_pose": val_mano_pose,
                "root_joint_cam": root_joint_cam,
                "do_flip": False
            }

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        # import ipdb; ipdb.set_trace()
        batch_size = len(batch_output)
        eval_result = [[], [], [], []]  # [mpjpe_list, pa-mpjpe_list]
        for n in range(batch_size):  # for each sample in the batch
            # gt
            data = copy.deepcopy(self.datalist[cur_sample_idx + n])
            output = batch_output[n]

            verts_out = output["pred_verts3d_w_gr"]
            joints_out = output["pred_joints3d_w_gr"]

            verts_gt, joints_gt = output["gt_verts3d_w_gr"], output["gt_joints3d_w_gr"]

            verts_gt -= joints_gt[self.root_joint_idx]
            joints_gt -= joints_gt[self.root_joint_idx]

            # root centered
            verts_out -= joints_out[self.root_joint_idx]
            joints_out -= joints_out[self.root_joint_idx]

            # root align
            gt_root_joint_cam = data["root_joint_cam"]
            verts_out += gt_root_joint_cam
            joints_out += gt_root_joint_cam

            verts_gt += gt_root_joint_cam
            joints_gt += gt_root_joint_cam

            # align predictions
            joints_out_aligned = align_w_scale(joints_gt, joints_out)
            verts_out_aligned = align_w_scale(verts_gt, verts_out)

            # m to mm
            joints_out *= 1000
            joints_out_aligned *= 1000
            joints_gt *= 1000
            verts_out *= 1000
            verts_out_aligned *= 1000
            verts_gt *= 1000

            eval_result[0].append(np.sqrt(np.sum((joints_out - joints_gt)**2, 1)).mean())
            eval_result[1].append(np.sqrt(np.sum((joints_out_aligned - joints_gt)**2, 1)).mean())
            eval_result[2].append(np.sqrt(np.sum((verts_out - verts_gt)**2, 1)).mean())
            eval_result[3].append(np.sqrt(np.sum((verts_out_aligned - verts_gt)**2, 1)).mean())

        MPJPE = np.mean(eval_result[0])
        PAMPJPE = np.mean(eval_result[1])
        MPVPE = np.mean(eval_result[2])
        PAMPVPE = np.mean(eval_result[3])
        score = PAMPJPE + PAMPVPE + MPJPE + MPVPE
        metric = {
            "MPJPE": MPJPE,
            "PAMPJPE": PAMPJPE,
            "MPVPE": MPVPE,
            "PAMPVPE": PAMPVPE,
            "score": score,
        }
        return metric
