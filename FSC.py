import json
import numpy as np
import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset
import os


MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]



class FSC147(Dataset):
    def __init__(self, 
                 split:str, 
                 subset_scale:float=1.0, 
                 resize_val:bool=True):
        """
        Parameters
        ----------
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']

        #!HARDCODED Dec 25: 
        self.data_dir = "data/FSC/"
        self.dataset_type = 'FSC_147'

        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir,self.dataset_type , f'annotation_{self.dataset_type}.json')
        self.data_split_file = os.path.join(self.data_dir,self.dataset_type ,f'Train_Test_Val_{self.dataset_type}.json')
        self.class_file = os.path.join(self.data_dir, self.dataset_type ,f'ImageClasses_{self.dataset_type}.txt')
        self.fsc147_d = os.path.join('FSC_147_D.json')
        self.split = split
        
        with open(self.data_split_file) as f:
            data_split = json.load(f)

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        self.idx_running_set = data_split[split]
        # subsample the dataset
        self.idx_running_set = self.idx_running_set[:int(subset_scale*len(self.idx_running_set))]

        self.class_dict = {}
        with open(self.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                # concat word as string
                val = ' '.join(val)
                self.class_dict[key] = val

        self.all_classes = list(set(self.class_dict.values()))

        random.shuffle(self.idx_running_set)

        
            
    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_id = self.idx_running_set[idx]
        anno = self.annotations[im_id]
        text = self.class_dict[im_id]

        bboxes = anno['box_examples_coordinates']
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            dots = np.array(anno['points'])

            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            image.load()
            Normalize = transforms.Compose([transforms.ToTensor()])
            image_pil = image.convert("RGB")
            image = Normalize(image)
            density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype('float32')   

            sample = {'image':image,'lines_boxes':rects,'gt_density':density, 'dots':dots, 'id':im_id}

            return sample['image'].float(), sample['gt_density'], rects,dots, image_pil, text ,im_id
        elif self.split == "test" or self.split == "test_coco" or self.split == "val_coco" or (self.split == "val" and not self.resize_val):
            dots = np.array(anno['points'])
            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            text = self.class_dict[im_id]
            image.load()
            W, H = image.size

            new_H = H #16*int(H/16)
            new_W = W #16*int(W/16)
            scale_factor = float(new_W)/ W
            # image = transforms.Resize((new_H, new_W))(image)
            Normalize = transforms.Compose([transforms.ToTensor()])
            image_pil = image.convert("RGB")
            image = Normalize(image)

            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0]*scale_factor)
                y1 = bbox[0][1]
                x2 = int(bbox[2][0]*scale_factor)
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            boxes = list()
            cnt = 0
            for box in rects:
                cnt+=1
                if cnt>3:
                    break
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:,y1:y2+1,x1:x2+1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())

            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)

            # Only for visualisation purpose, no need for ground truth density map indeed.
            gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')
            for i in range(dots.shape[0]):
                gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
            gt_map = torch.from_numpy(gt_map)
            gt_map = gt_map
            
            sample = {'image':image,'dots':dots, 'boxes':boxes, 'pos':rects, 'gt_map':gt_map}
            # return sample['image'].float(), sample['gt_map'], sample['boxes'], sample['pos'], text,image_pil
            return sample['image'].float(), sample['gt_map'], im_id, sample['pos'], text,image_pil
