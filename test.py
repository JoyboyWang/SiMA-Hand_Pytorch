import argparse
import os
import torch
import json
from tqdm import tqdm
from data_loader.data_loader import fetch_dataloader
from model.fetch_model import fetch_model
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--resume", default=None, type=str, help="Path of model weights")

def test(model, mng: Manager):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # Compute metrics over the dataset
        for split in ["val", "test"]:
            if split not in mng.dataloader:
                continue
            # Initialize loss and metric statuses
            mng.reset_loss_status()
            mng.reset_metric_status(split)
            # Use tqdm for progress bar
            t = tqdm(total=len(mng.dataloader[split]))
            cur_sample_idx = 0
            for batch_idx, batch_input in enumerate(mng.dataloader[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input)
                # Compute model output
                batch_output = model(batch_input)
                # Get real batch size
                if "img" in batch_input:
                    batch_size = batch_input["img"].size()[0]
                elif "img_0" in batch_input:
                    batch_size = batch_input["img_0"].size()[0]
                else:
                    batch_size = mng.cfg.test.batch_size
                    
                batch_output = tool.tensor_gpu(batch_output, check_on=False)
                batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                # evaluate
                metric = mng.dataset[split].evaluate(batch_output, cur_sample_idx)
                cur_sample_idx += len(batch_output)
                mng.update_metric_status(metric, split, batch_size)

                # Tqdm settings
                t.set_description(desc="")
                t.update()

            mng.print_metric(split, only_best=False)
            t.close()


def main(cfg):
    # Set rank and is_master flag
    cfg.base.only_weights = False
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.base.model_dir, "test.log"))
    # Print GPU ids
    gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    logger.info("Using GPU ids: [{}]".format(gpu_ids))
    # Fetch dataloader
    cfg.data.eval_type = ["test"]
    dl, ds = fetch_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg.model.name, cfg)
    # Initialize manager
    mng = Manager(model=model, optimizer=None, scheduler=None, cfg=cfg, dataloader=dl, dataset=ds, logger=logger)
    # Test the model
    mng.logger.info("Starting test.")
    # Load weights from restore_file if specified
    if mng.cfg.base.resume is not None:
        mng.load_ckpt()
    test(model, mng)


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg
    # Update args into cfg.base
    cfg.base.update(vars(args))

    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    main(cfg)
