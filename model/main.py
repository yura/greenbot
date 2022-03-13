'''
Examples:
python3 main.py --epochs 5 --dataset_path ./datasets/seven_plastics_small
python3 main.py --mode test --dataset_path ./datasets/seven_plastics_small --checkpoint ./weights/MobileNetV3_2022-03-12_09-36-15_224_8cl_e100_acc0.8495_seven_plastics.pth
python3 main.py --mode predict --img_path ./datasets/seven_plastics/3_polyvinylchloride_PVC/IMG_6443.jpg --checkpoint ./weights/MobileNetV3_2022-03-12_09-36-15_224_8cl_e100_acc0.8495_seven_plastics.pth
'''


from loader import *
from config import *
import torch
from torchvision import models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from time import time, localtime, strftime
import copy
from os.path import basename

CLASS_NAMES = {'1_polyethylene_PET': 0, '2_high_density_polyethylene_PE-HD': 1, '3_polyvinylchloride_PVC': 2, '4_low_density_polyethylene_PE-LD': 3, '5_polypropylene_PP': 4, '6_polystyrene_PS': 5, '7_other_resins': 6, '8_no_plastic': 7}

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # compute prediction error
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, y)
        
        # backprop
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)
        
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size
    
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            
            loss = loss_fn(outputs, y)
            
            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
            
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size
    print(f"Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    return epoch_acc
    
def predict(cfg, model, device):
    transform = T.Compose([
                T.Resize(cfg.img_size, Image.ANTIALIAS),
                T.CenterCrop(cfg.img_size),
                T.ToTensor(),
                T.Normalize(cfg.norm_mean, cfg.norm_std)
            ])
    X = Image.open(cfg.img_path)
    X = transform(X)
    X = torch.unsqueeze(X, 0)
    X = X.to(device)
    with torch.no_grad():
        outputs = model(X)
    predicted_class = outputs[0].argmax(0)
    
    return predicted_class

def main(cfg):
    run_datetime = localtime()
    print(f"Start at {strftime('%Y-%m-%d_%H-%M-%S',run_datetime)}")
    cfg.norm_mean, cfg.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # get data from dataset
    plastic_dataset = SevenPlastics(cfg)
    if cfg.mode == "train":
#        train_data = SevenPlastics(cfg)
#        test_data = SevenPlastics(cfg, is_train=False)
#        train_count = int(0.7 * len(train_data))
#        test_count = int(0.3 * len(test_data))
#
#        # create splitted datasets based on whole data
#        train_dataset, _ = random_split(train_data, (train_count, len(train_data)-train_count))
#        test_dataset, _ = random_split(test_data, (test_count, len(test_data)-test_count))

        train_length=int(0.7 * len(plastic_dataset))
        test_length=len(plastic_dataset)-train_length
        
        train_dataset,test_dataset = random_split(plastic_dataset,(train_length,test_length))
        
        # create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)

        print(f"train/test dataloader length: {len(train_dataloader.dataset)}/{len(test_dataloader.dataset)}")
        print(f"train/test dataloader batches: {len(train_dataloader)}/{len(test_dataloader)}")
    # for test and predict
    elif cfg.mode == "test":
        test_dataloader = DataLoader(plastic_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
        print(f"test dataloader length: {len(test_dataloader.dataset)}")
        print(f"test dataloader batches: {len(test_dataloader)}")
    
    # load weights
    if cfg.mode == "train" and not cfg.checkpoint:
        model = models.mobilenet_v3_large(pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False)
        
        # freeze all the parameters in the network
#        for param in model.parameters():
#            param.requires_grad = False
            
        # finetuning the convnet
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features,  out_features=len(plastic_dataset.class_map))
        
    elif cfg.checkpoint:
        print(f"loading model from checkpoint...")
        model = models.mobilenet_v3_large(width_mult=1.0,  reduced_tail=False, dilated=False)
        print(f"elif cfg.checkpoint: plastic_dataset.class_map): {plastic_dataset.class_map}")
        # finetuning the convnet
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=len(CLASS_NAMES))
        model.load_state_dict(torch.load(cfg.checkpoint, map_location=torch.device(device)))
    else:
        raise NotImplementedError(f"Set [--checkpoint] argument value for model testing. Current value: checkpoint={cfg.checkpoint}")

    model = model.to(device)

    # criterion
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
#    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4, eps=0.0316, alpha=0.9)
    
    # decay LR by a factor 0.1 every 5 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    since = time()
    predicted_classes_dict = {}
    epoch_acc = 0.0
    best_acc = 0.0

    if not cfg.mode == 'train':
        cfg.epochs = 1
        
    for t in range(cfg.epochs):
    
        if cfg.mode == "train":
            print(f"\nEpoch {t+1}/{cfg.epochs}\n{'-'*20}")
            train(train_dataloader, model, loss_fn, optimizer, device)
            exp_lr_scheduler.step()
        
        if cfg.mode in ["test", "train"]:
            epoch_acc = test(test_dataloader, model, loss_fn, device)
        
        if cfg.mode == "predict":
            predicted_class_id = predict(cfg, model, device)
            print(f"Predicted class: {list(CLASS_NAMES.keys())[predicted_class_id.item()]}")
            
        # deep copy the best model
        if cfg.mode == "train" and epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(model.state_dict())
          print(f"Best model with accuracy: {best_acc:.4f}")
        
    time_elapsed = time() - since
    print(f"\n{cfg.mode.capitalize()} is done in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s")
    
    if cfg.mode == 'train':
        # save best model
        model_file_name = f"./weights/{model.__class__.__name__}_{strftime('%Y-%m-%d_%H-%M-%S', run_datetime)}_{cfg.img_size}_{len(plastic_dataset.class_map)}cl_e{cfg.epochs}_acc{best_acc:.4f}_{basename(cfg.dataset_path)}.pth"
        torch.save(best_model_wts, model_file_name)
        print(f"Saved PyTorch best model state to {model_file_name}")

if __name__ == '__main__':
  cfg = get_args()
  main(cfg)
