from loader import *
from config import *
import torch
from torchvision import models
from torch.utils.data import DataLoader, random_split
    
def main(cfg):
    cfg.norm_mean, cfg.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    if cfg.mode == "train":
        train_data = SevenPlastics(cfg)
        test_data = SevenPlastics(cfg, is_train=False)
        train_count = int(0.7 * len(train_data))
        test_count = int(0.3 * len(test_data))

        # create splitted datasets based on whole data
        train_dataset, _ = random_split(train_data, (train_count, len(train_data)-train_count))
        test_dataset, _ = random_split(test_data, (test_count, len(test_data)-test_count))

        # create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
        print(f"train/test dataloader length: {len(train_dataloader.dataset)}/{len(test_dataloader.dataset)}")
        print(f"train/test dataloader batches: {len(train_dataloader)}/{len(test_dataloader)}")

if __name__ == '__main__':
  cfg = get_args()
  main(cfg)
