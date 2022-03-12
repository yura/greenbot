import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset

class SevenPlastics(Dataset):
    def __init__(self, cfg, is_train=True):
        self.dataset_path = cfg.dataset_path
        self.is_train = is_train
        self.class_map = {}
        # load dataset
        self.x, self.y = self.get_data()
        # set transforms
        if self.is_train:
            self.tranform = T.Compose([
                T.Resize(cfg.img_size, Image.ANTIALIAS),
                T.RandomRotation(180),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
        else:
            self.tranform = T.Compose([
                T.Resize(cfg.img_size, Image.ANTIALIAS),
                T.ToTensor()
            ])
        self.normalize = T.Compose([T.Normalize(cfg.norm_mean, cfg.norm_std)])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x)
        x = self.normalize(self.tranform(x))
        
        return x, y
    
    def get_data(self):
        x, y = [], []
    
        class_id = 1
        for class_name in sorted(os.listdir(self.dataset_path)):
            self.class_map[class_name] = class_id
            img_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(img_path):
                for img_name in os.listdir(img_path):
                    x.append(os.path.join(img_path, img_name))
                    y.append(class_id)
            class_id += 1
        print(f"self.class_map: {self.class_map}")

        assert len(x) == len(y), "Number of x and y should be same!"

        return x, y
