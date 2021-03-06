import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GreenBot")
    parser.add_argument("--dataset_path", default="./datasets/seven_plastics", type=str,
                        help="path to dataset dir")
    parser.add_argument("--mode", default="train", type=str,
                        help="set script mode {train, test, predict} (default: train)")
    parser.add_argument("--img_size", default=224, type=int,
                        help="image resize dimensions (default: 224")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="train batch size (default: 8)")
    parser.add_argument("--workers", default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--checkpoint", default="", type=str,
                        help="file with saved checkpoint")
    parser.add_argument("--img_path", default="./datasets/seven_plastics/AORA7148.jpg", type=str,
                        help="single image path (for prediction)")
                        
    args = parser.parse_args()
    
    return args

