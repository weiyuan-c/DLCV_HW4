import argparse
import os
import random
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from models import Classifier

# setting seed
seed = 72373981
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ImgDataset(Dataset):
    def __init__(self, root, csv_file, tfm=None):
        super(ImgDataset).__init__()
        self.root = root
        data = pd.read_csv(csv_file)
        self.fnames = data['filename']
        self.tfm = tfm

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        fname = os.path.join(self.root, fname)
        img = Image.open(fname).convert('RGB')
        img = self.tfm(img)

        return img

class_name = ['Couch', 'Helmet', 'Refrigerator', 'Alarm_Clock', 'Bike', 'Bottle', 'Calculator', 
              'Chair', 'Mouse', 'Monitor', 'Table', 'Pen', 'Pencil', 'Flowers', 'Shelf',
              'Laptop', 'Speaker', 'Sneakers', 'Printer', 'Calendar', 'Bed', 'Knives', 'Backpack', 
              'Paper_Clip', 'Candles', 'Soda', 'Clipboards', 'Fork', 'Exit_Sign', 'Lamp_Shade', 
              'Trash_Can', 'Computer', 'Scissors', 'Webcam', 'Sink', 'Postit_Notes', 'Glasses', 
              'File_Cabinet', 'Radio', 'Bucket', 'Drill', 'Desk_Lamp', 'Toys', 'Keyboard', 'Notebook', 
              'Ruler', 'ToothBrush', 'Mop', 'Flipflops', 'Oven', 'TV', 'Eraser', 'Telephone', 'Kettle', 
              'Curtains', 'Mug', 'Fan', 'Push_Pin', 'Batteries', 'Pan', 'Marker', 'Spoon', 'Screwdriver', 
              'Hammer', 'Folder']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('-c', '--csv_dir', type=str, default='hw4_data/office/val.csv')
    parser.add_argument('-d', '--data_dir', type=str, default='hw4_data/office/val')
    parser.add_argument('--pred', type=str, default='pred.csv')

    parser.add_argument('--weight', type=str, default='best_model.pth')
    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    cfg = parser.parse_args()

    # data transfomation
    test_tfm = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)), 
        T.ToTensor(), 
        T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # load model
    resnet = models.resnet50(weights=None).to(device)
    model = Classifier(resnet).to(device)
    model.load_state_dict(torch.load(cfg.weight, map_location=device))
    # dataset
    val_set = ImgDataset(cfg.data_dir, cfg.csv_dir, tfm=test_tfm)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    # validation 
    model.eval()
    val_pred = []
    for i, img in enumerate(tqdm(val_loader)):
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            pred = list(output.argmax(dim=1).squeeze().detach().cpu().numpy())
        val_pred += pred
    
    ori_label = []
    for ele in val_pred:
        ori_label.append(str(class_name[ele]))

    df = pd.DataFrame()
    data = pd.read_csv(cfg.csv_dir)
    df['id'] = data['id']
    df['filename'] = data['filename']
    df['label'] = ori_label
    df.to_csv(cfg.pred, index=False)