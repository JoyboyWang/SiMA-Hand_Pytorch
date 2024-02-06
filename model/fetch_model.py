import torch
from torch.nn.parallel import DataParallel as DP
from model.model import *

def fetch_model(model_name, cfg):
    if model_name == "svr_hand":
        model = SVRHand(cfg)
    else:
        raise NotImplementedError

    if cfg.base.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)

        model = model.cuda()
        model = DP(model)

    return model