import os
import time
import numpy as np
# import tensorflow as tf
import random
from threading import Thread
from scipy.io import loadmat
from skvideo.io import vread
import pdb
import torch
from torch.utils.data import Dataset
import pickle
import cv2

import glob 
from pathlib import Path 
import shutil as sh 
import json
from torchvision import transforms
from tubes_mama import extract_tubes

class TinyActionsLoader(Dataset):
    #same signature as original ucf101 datalaoder
    def __init__(self, name, clip_shape, batch_size,num_activities = 36,num_frames = 8, tube_len=100, collate_batch = 12, load_file_fp='empty', percent='40', percent_vids='20', use_random_start_frame=False, transform=None):
        
        self.tube_len = tube_len
        self.local_data_path = Path('.').absolute()
        self.data_dir_rajat = Path('/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/total_data')
        self.data_dir_emily = Path('home/em585489/mama_dataloader/tubes')
        # glob.glob 
        self.data_dir = glob.glob('/home/em585489/mama_dataloader/tubes/**/*.avi', recursive=True) # can use sorted() 
        #read the dictionary 
        cat_to_int_path = '/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/to_release/categories.txt' #categories.txt
        self.cat_to_int = {}
        d= {}
        with open(cat_to_int_path) as f:
            lines = f.readlines()
            #print("before reading lines",lines)
            lines = [line.strip().split(' ') for line in lines]
            for line in lines:
                #print("line",line)
                d[line[1]] = int(line[0])
        self.cat_to_int = d  
        # print('this is categories', self.cat_to_int)

        if name == 'train':
        #   train_txt = Path('/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/train.txt')
        #   self.vid_files = self.load_video_ids(train_txt) #train  txt.
          train_pkl  = "/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/sampled_train.pkl"
          self.vid_files = self.create_sampled_list(train_pkl,name)
        #   self.vid_files = sorted(self.create_sampled_list(train_pkl,name))
          self.vid_files = self.vid_files[:12000]
          self.shuffle = True
          self.name = 'train'
        else:
        #   test_txt = Path('/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/test.txt')
        #   self.vid_files = self.load_video_ids(test_txt) #test txt/
          test_pkl = "/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/sampled_test.pkl"
          self.vid_files = self.create_sampled_list(test_pkl,name)
          pick = 100
          self.vid_files = self.vid_files[:pick]
          self.shuffle = False
          self.name = 'test'

        # print("self, len of vid_files",len(self.vid_files))
        self.num_frames = num_frames
        self.num_activities = num_activities
        self._use_random_start_frame = use_random_start_frame
        self._height = clip_shape[0]
        self._width = clip_shape[1]
        #self._channels = channels
        self._batch_size = batch_size
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def create_sampled_list(self, pkl_path, name):
        dbfile = open(pkl_path, 'rb')     
        counter  = pickle.load(dbfile)
        if name =='train':
            thresh = 1500 
        else:
            thresh = 1000
        picked = set() #set of all picked ids 
        for activity in counter.keys():
            ids = counter[activity]
            if len(ids) <thresh:
                #put all  thhe ids in the picked 
                #print("trivial actovoty",activity)
                for id in ids:
                    picked.add(id)
            else:
                #generate the         
                sampled_ids = random.sample(ids, thresh)
                #print("non trivial actovoty",activity)

                #print("sampled ids are", len(sampled_ids))
                for id in sampled_ids:
                    picked.add(id)

        return list(picked)

    def load_video_ids(self,txt_path):
        ids = set()
        with open(str(txt_path)) as f:
            lines = f.readlines()
            #print("before reading lines",lines)
            lines = [line.strip().split(' ') for line in lines]
            for line in lines:
                # print(line)
                ids.add(line[0])
        #filter ids 
        stealing_path = '/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/stealing_hex_codes.txt'
        with open(stealing_path) as f:
            drop_lines = f.readlines()
            drop_lines = [drop_line.strip() for drop_line in drop_lines]
        # drop_lines = set(drop_lines)
        #read the dropped files 
        for el in drop_lines:
            #print("el",el)
            if el in ids:
                ids.remove(el)
        ids = list(ids)
        return sorted(ids)

    
    # loads the video along with the annotations 
    def load_video(self,v_id,index):
        # print('starting load_video')
        # vid_dir = self.data_dir_emily/'34'/v_id
        # print('this is the index', index)
        vid_path = self.data_dir[index]    # will be .avi
        # vid_dir = self.data_dir_rajat/v_id
        # vid_path = vid_dir/(v_id + '.mp4')
        # ann_path = self.data_dir_rajat/v_id/(v_id + '.json')
        # ann_path = vid_dir/(v_id + '.json')

        path_to_list = vid_path.split("/")
        # print('this is the path spilt', path_to_list)
        
        label_id = path_to_list[5]

        if label_id == str(v_id):
            label_id = path_to_list[6]     # dont think i need this because the label will always be in the 5th place

        label_id = int(label_id)
        
        #read video 
        cap = cv2.VideoCapture(str(vid_path))
        video = []
        success= True
        while success:
            success, image = cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.uint8)
                # print("image shape",image.shape)
                video.append(image)
        if video==[]:
            print('Error:', str(vid_path))
            return None, None

        video = np.stack(video,0)
        # # print("video shape is",video.shape)
        # n_frames, h, w, ch = video.shape
        # bbox = np.zeros((self.num_activities, n_frames, h, w, 1), dtype=np.uint8) #multi activity features are present 
        # resized_bbox = np.zeros((self.num_activities, n_frames, resize_h, resize_w, 1), dtype=np.uint8)
        # label = set() #multiple activities can occur in a video 
        # multi_frame_annot = [] # a list of all the frames which are annotated

        # #read annotation 
        # with open(str(ann_path),'r') as f:
        #     ann = json.load(f)
        # # print("keys",ann.keys())
        # f_ids = list(ann.keys())
        # print("fids",f_ids)
        # print(list(label))
        # return video, bbox, list(label), multi_frame_annot
        # return None, None, None , None


        if video.shape[0] == 150:
            video = video[:100, :, :, :]

        # if video.shape[0] == 150 or video.shape[0] == 100:    making it less frames 
        #     video = video[:50, :, :, :]

        # print('this is video and label_id', video.shape, label_id)
                    
        return video, label_id



        # return video, label_id
    
    ########################### old load_vid ################################################

    # def generate_envelope(self, bbox):
    #     env = [float('inf'), float('inf'), -float('inf'), -float('inf')]
    #     for bounding_box in bbox:
    #         x1, x2, y1, y2 = bounding_box
    #         env[0] = min(x1, env[0])
    #         env[1] = min(y1, env[1])
    #         env[2] = max(x2, env[2])
    #         env[3] = max(y2, env[3])

    #     return env

    def get_vid_id(self, index):
        return self.vid_files[index]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

        # return len(self.vid_files)

    def __getitem__(self, index):
        # print("*********************************",index)
        depth = 8    # self.num_frames
        video_rgb = np.zeros((depth, self._height, self._width, 3))
        label_cls = np.zeros((self.num_activities,depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        mask_cls = np.zeros((self.num_activities,depth, self._height, self._width, 1))
        
        resize_h,resize_w =256,256 #frames are resized to this before taking a crop. this improves the probability that a particular detection lies in the crop. 
        
        ############ Getting the ann_path #############
        v_id = self.vid_files[index]
        # print('this is the v_id', v_id)
        data_dir = '/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/total_data'
        vid_dir = data_dir + '/' + v_id
        # print(vid_dir)
        ann_path = vid_dir + '/ann_with_actor_id.json'
        # ann_path = vid_dir + '/' + v_id + '.json'
        # print('this is the ann_path', ann_path)

        # vid_dir = self.data_dir/v_id
        # ann_path = vid_dir/(v_id + '.json')
        ###############################################

        # clip, bbox_clip, label, annot_frames, resized_bbox_clip = self.load_video(v_id, index, resize_h,resize_w)
        clip, label = self.load_video(v_id, index)
        temp = np.zeros((3, 112, 112, 100))
        temp = torch.from_numpy(temp)
        if clip is None or label is None:
            return [temp, 1]
        # print('this is the label', label)
        
        # if label in self.cat_to_int.keys():
        #     label_id = self.cat_to_int[label]
        # print('type of annot frames: ', type(annot_frames), 'elem of the annot frames: ', len(annot_frames))
        # clip -> num_frames x H x W x CH
        # print('clip shape before: ', clip.shape)
        # print("bbox clip shape: ", bbox_clip.shape)  # ? x # x # x # x CH  -> ? - num of activities 
        # clip=None
        # if clip is None:
        #     video_rgb = np.zeros((depth, 224,224, 3))
        #     label_cls = np.zeros((self.num_activities,depth, 224,224, 1))     # FG/BG or actor (person only) for this dataset
        #     mask_cls = np.zeros((self.num_activities,depth, 224,224, 1))

        #     video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        #     video_rgb = torch.from_numpy(video_rgb)            
        #     label_cls = np.transpose(label_cls, [4, 0, 1, 2,3])
        #     label_cls = np.squeeze(label_cls,0)
        #     label_cls = torch.from_numpy(label_cls)
        #     mask_cls = np.transpose(mask_cls, [4, 0, 1, 2,3])
        #     mask_cls = np.squeeze(mask_cls,0)

        #     mask_cls = torch.from_numpy(mask_cls)
        #     # print("in noen case",video_rgb.shape,label_cls.shape,mask_cls.shape)
        #     slice_len =  1
        #     slice_len = torch.Tensor(slice_len)
        #     action_tensor = [-1 for _ in range(36)]
            
        #     print("hare krishna 1",video_rgb.shape,torch.Tensor(action_tensor).shape,label_cls.shape,mask_cls.shape,)

        #     sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor(action_tensor),'mask_cls':mask_cls}
        #     return sample
            

        # print('shape of clip', clip.shape)
        tube_output = []
        clip = torch.from_numpy(clip)
        # print('clip shape before swapaxes', clip.shape)
        clip = torch.swapaxes(clip, 0, 3)
        # print('clip shape after swapaxes', clip.shape)
        tube_output.append(clip)
        tube_output.append(label)
        # print('type of clip', type(clip))
        # print('type of label', type(label))


        # tube_output = extract_tubes(clip, v_id, label_ids=self.cat_to_int, tube_len=self.tube_len)
        # print('this is the tube_outputl, should be tensor: label', type(tube_output.keys()))

        # crops = np.stack(crops)
        # crop_tensor = tf.convert_to_tensor(crops)
        # print('this it crop_tensor shape: ', crop_tensor.shape)

        
        # print('reached the end of __getitem__')
        return tube_output  
        # used to be return sample



#main function 
if __name__ == '__main__':
    import imageio 
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 5
    seed = 1800
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataloader = TinyActionsLoader(name, clip_shape, batch_size, use_random_start_frame=False, load_file_fp='training_annots_multi_5per_prune1wENT_total5per_interp.pkl')
    # print("datalaoder intiialized")
    # print("len of dataloader",len(dataloader))
    # print('starting the get item')
    # for i in range(len(dataloader)):
    #     sample = dataloader[i]
    # sample = dataloader[2]
    # print('this should be type dictionary', type(sample))
