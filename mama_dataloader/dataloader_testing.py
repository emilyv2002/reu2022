from ctypes import resize
import os
import time
import numpy as np
import random
from threading import Thread
# from scipy.io import loadmat
# from skvideo.io import vread
import pdb
import torch
from torch.utils.data import Dataset
import pickle
import cv2

import glob 
from pathlib import Path 
import shutil as sh 
import json

from test_tubes import extract_tubes

#just return the frames without any cropping
#read them and resize to 224,224. including bboxes 
# check once by visualizing the dataloader 
#return a list of valid frames
class TinyActionsLoaderTesting(Dataset):
    #same signature as original ucf101 datalaoder
    def __init__(self, name, clip_shape, batch_size,num_activities = 36,num_frames = 8, load_file_fp='empty', percent='40', percent_vids='20', use_random_start_frame=False):
        
        self.local_data_path = Path('.').absolute()
        self.data_dir_rajat = Path('/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/to_release/test')   # change it to to_release/test, used to be total_data
        self.data_dir = glob.glob('/home/em585489/mama_dataloader/tubes_test/**/*.avi', recursive=True)
        #read the dictionary 
        cat_to_int_path = '/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/to_release/categories.txt'
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

        if name == 'test':
          test_txt = Path('/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/test.txt')
          self.vid_files = self.load_video_ids(test_txt) #test txt/
          self.vid_files = self.vid_files[:]
          import random 
          random.shuffle(self.vid_files)
        #   pick = 100
        #   self.vid_files = self.vid_files[:pick]
          self.shuffle = False
          self.name = 'test'

        else:
          raise Exception("Prabhu, this is test loader!!!, enter 'test' please")
          pass
        # print("self",len(self.vid_files))
        self.num_frames = num_frames
        self.num_activities = num_activities
        self._use_random_start_frame = use_random_start_frame
        self._height = clip_shape[0]
        self._width = clip_shape[1]
        #self._channels = channels
        self._batch_size = batch_size
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)

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

    #loads the video along with the annotations 
    def load_video(self,v_id,index, resize_h,resize_w):
        vid_dir = self.data_dir_rajat/v_id
        # vid_path = vid_dir/(v_id + '.mp4')
        ann_path = vid_dir/(v_id + '.json')

        # print('this is the index', index)

        vid_path = self.data_dir[index]
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
        # print("video shape is",video.shape)
        # n_frames, h, w, ch = video.shape
        # bbox = np.zeros((self.num_activities, n_frames, h, w, 1), dtype=np.uint8) #multi activity features are present 
        # resized_bbox = np.zeros((self.num_activities, n_frames, resize_h, resize_w, 1), dtype=np.uint8)
        # label = set() #multiple activities can occur in a video 
        # multi_frame_annot = [] # a list of all the frames which are annotated

        #read annotation 
        # with open(str(ann_path),'r') as f:
        #     ann = json.load(f)
        # # print("keys",ann.keys())
        # f_ids = list(ann.keys())
        # print("fids",f_ids)
        # print(list(label))
        # return video, bbox, list(label), multi_frame_annot
        # return None, None, None , None
        # for i,f_id in enumerate(f_ids):
        #     # f_id = int(f_id)
        #     # print("reading gframe",i,"out of",len(f_ids))
        #     if len(list(ann[f_id].keys()))!=0:
        #         multi_frame_annot.append(int(f_id))
        #     for n_activities,activity in enumerate(ann[f_id].keys()):
        #         # print("ac",n_activities, "out of",len(ann[f_id]))
        #         label.add(self.cat_to_int[activity])
                
        #         for detection in ann[str(f_id)][activity]:
        #             #in the current frame, with activity, there can be several possible detections taking place. 
                    
        #             try:
        #                 activity_id = self.cat_to_int[activity]
        #                 x1,y1,x2,y2 = detection
        #                 # print("detection",detection)
        #                 f_id = int(f_id)
        #                 #append the detection to the bbox
        #                 # print("x",x1,y1,x2,y2)
                        
        #                 # print("aid",activity_id,type(activity_id))
        #                 # print("type",type(bbox[activity_id][f_id]))
        #                 #print("shape",bbox[activity_id][f_id].shape)
        #                 bbox[activity_id][f_id] = cv2.rectangle(bbox[activity_id][f_id], (x1,y1),(x2,y2), (1,1,1), -1)

        #                 # bbox[activity_id][f_id] = cv2.rectangle(bbox[activity_id][f_id], (y1,x1),(y2,x2), (1,1,1), -1)
        #                 bbox[activity_id][f_id] = (bbox[activity_id][f_id]>0)*1
        #                 #print("unique",np.unique(bbox[activity_id][f_id]))
                        
        #                 bbox[activity_id][f_id] = np.ascontiguousarray(bbox[activity_id][f_id], dtype=np.uint8)
        #                 # print("resize height",resize_h,resize_w)
        #                 # exit(1)
        #                 #resized operation
        #                 r_x1,r_y1,r_x2,r_y2 = x1,y1,x2,y2
        #                 # print(r_x1,r_y1,r_x2,r_y2)
        #                 r_x1 = int((x1/w)*resize_h)
        #                 r_y1= int((y1/h)*resize_w)
        #                 r_x2 = int((x2/w)*resize_h)
        #                 r_y2 = int((y2/h)*resize_w)
        #                 # print(r_x1,r_y1,r_x2,r_y2)
        #                 # exit(1)
                        
        #                 resized_bbox[activity_id][f_id] = cv2.rectangle(resized_bbox[activity_id][f_id], (r_x1,r_y1),(r_x2,r_y2), (1,1,1), -1)
        #                 resized_bbox[activity_id][f_id] = (resized_bbox[activity_id][f_id]>0)*1
        #                 # print("unique",np.unique(resized_bbox[activity_id][f_id]))
                        
        #                 resized_bbox[activity_id][f_id] = np.ascontiguousarray(resized_bbox[activity_id][f_id], dtype=np.uint8)
        #             except:
        #                 print("error in annotating bbox")
                #cv2.imwrite('./test.jpg', bbox[activity_id][f_id]*255)
                #exit(1)
        # print("unique",np.unique(bbox))
        # return video, bbox, list(label),multi_frame_annot,resized_bbox
        return video, label_id
            
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

        # return len(self.vid_files)

    def __getitem__(self, index):
        # print("*********************************",index)

        vid_id = self.vid_files[index]
        clip, label = self.load_video(vid_id, index, 112, 112)

        tube_output = []
        clip = torch.from_numpy(clip)
        clip = torch.swapaxes(clip, 0, 3)
        tube_output.append(clip)
        tube_output.append(label)

        return tube_output
        
        # try:
        #     resize_h,resize_w =224,224 #frames are resized to this before taking a crop. this improves the probability that a particular detection lies in the crop.  
        #     v_id = self.vid_files[index]
        #     # v_id =  "0xd50e81f1"
        #     # clip, bbox_clip, label, annot_frames,resized_bbox_clip = self.load_video(v_id,index,resize_h,resize_w)
        #     clip, label = self.load_video(v_id,index,resize_h,resize_w)

        #     # tube_output = extract_tubes(clip, v_id, label_ids=self.cat_to_int, tube_len=100)

        #     # print('shape of clip', clip.shape)
        #     tube_output = []
        #     clip = torch.from_numpy(clip)
        #     # print('clip shape before swapaxes', clip.shape)
        #     clip = torch.swapaxes(clip, 0, 3)
        #     # print('clip shape after swapaxes', clip.shape)
        #     tube_output.append(clip)
        #     tube_output.append(label)
            # print('type of clip', type(clip))
            # print('type of label', type(label))
            
            #print("before",clip.shape, resized_bbox_clip.shape)
            # n_frames = clip.shape[0]
            # depth = n_frames
            
            # video_rgb = np.zeros((depth, self._height, self._width, 3)) 
            # #label cls will contain segmentation volume
            
            # #print("after video_rgb.shape",video_rgb.shape)
            # for j in range(depth):
            #     img = clip[j]
            #     img = cv2.resize(img, (224,224),interpolation=cv2.INTER_AREA )
            #     img = img/255
            #     video_rgb[j]= img
            #     #print("iomg",img)
            #     #exit(1)
            # #exit(1)
            # #print("after completion",video_rgb.shape, resized_bbox_clip.shape)
            # video_rgb = np.transpose(video_rgb,(3,0,1,2)) #channel first 
            # resized_bbox_clip = np.squeeze(resized_bbox_clip,4)
            # while len(label)<36:
            #     label.append(-1)
            # #print("after completion",video_rgb.shape, resized_bbox_clip.shape,label)
            # video_rgb = torch.from_numpy(video_rgb)
            # resized_bbox_clip = torch.from_numpy(resized_bbox_clip)

            # return v_id,video_rgb, resized_bbox_clip, torch.Tensor(label)
        #     return tube_output

        # except:
        #     depth = 8
        #     print("some error occured")
            # video_rgb = np.zeros((depth, self._height, self._width, 3)) 
            # resized_bbox_clip = np.zeros((self.num_activities,depth,self._height, self._width))
            # label = [-1 for _ in range(36)]
            # return v_id,video_rgb, resized_bbox_clip, torch.Tensor(label) #0 stands for bg 

        # def generate_video(frames, masks):
            
        #     #print("in video generation",frames.shape,masks.shape)
        #     # exit(1)
        #     n_frames,h,w, _ = frames.shape
        #     video_dump_path = str(index) + '.avi'
        #     video = cv2.VideoWriter(video_dump_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
            
        #     for frame_id in range(n_frames):

        #         # print("frames shape",frames.shape)
        #         frame = frames[frame_id,:,:,:]

        #         mask= masks[:,frame_id,:,:] #3 dimensions 
                
                
        #         mask = np.sum(mask, 0)
        #         # print("viz",frame.shape,mask.shape)
        #         # exit(1)
                
        #         # mask = np.transpose(mask,(1,0))
        #         # print("frame",frame.shape,mask.shape)
        #         # exit(1)

        #         mask = (mask >0)*1
        #         #print("mask shape",mask.shape)
        #         # exit(1)
        #         mask = np.expand_dims(mask, 2)
        #         def mask_over_image(image,mask,alpha=0.3):
        #             return alpha*mask + (1-alpha)*image
                
        #         def NormalizeData(data):
        #             return (data - np.min(data)) / (np.max(data) - np.min(data))
        #         # print(frame)
        #         # exit(1)
        #         frame = NormalizeData(frame)
        #         # print(frame)
        #         # exit(1)
        #         # mask =  NormalizeData(mask)
        #         vis_frame = mask_over_image(frame,mask)
        #         # print("vis frame",vis_frame)
                
        #         vis_frame = NormalizeData(vis_frame)
        #         # print("vis frame",vis_frame)
        #         vis_frame = np.uint8(vis_frame*255)
        #         # print("unique vis_frame",vis_frame)
        #         # exit(1)
        #         # vis_frame = np.uint8(frame*255)
        #         # print(vis_frame)
        #         # print("vis frame",np.min(vis_frame),np.max(vis_frame),np.unique(mask))
        #         path = str(index)+"_"+str(frame_id)+'.jpg'
        #         #print("writing image",path,vis_frame.shape)
        #         cv2.imwrite(path,vis_frame)
        #         video.write(vis_frame)
        #         # exit(1)
        #     video.release()
        
        # generate_video(video_rgb,resized_bbox_clip)
        
        
        # generate_video(video_rgb,bbox_clip)
        #generate_video(video_rgb,label_cls)
        
        

#main function 
if __name__ == '__main__':
    import imageio 
    from torch.utils.data import DataLoader

    
    name='test'
    clip_shape=[224,224]
    channels=3
    batch_size = 5
    seed = 402
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataloader = TinyActionsLoader(name, clip_shape, batch_size, use_random_start_frame=False, load_file_fp='training_annots_multi_5per_prune1wENT_total5per_interp.pkl')
    print("datalaoder intiialized")
    print("len of datalaoder",len(dataloader))
    for i in range(len(dataloader)):
        sample = dataloader[i]
    # sample = dataloader[10]
    # val_data_loader = DataLoader(
    #         dataset=dataloader,
    #         batch_size=1,
    #         num_workers=4,
    #         shuffle=False
    #     )

    # for idx, sample in enumerate(val_data_loader):
    #     print("done idx",idx)
