'''
Tube Extraction for UCF-MAMA: Testing Videos
Created by Emily Vo
'''
from ast import Dict
import os, sys
import json
import torch
import numpy as np
import pickle
import cv2
import pickle
from pathlib import Path 

from pathlib import Path
import random
# from cvpr_dataset_loader import TinyActionsLoader

# TinyActionsLoader >> just training 
# tubes for all 

################# Initialize ###################################

data_dir = '/home/rmodi/crcv_work/tiny_dataset_creation/final_dataset/tiny_detections_cvpr_2022/to_release/test'
dump_path = '/home/em585489/mama_dataloader/tubes_test'
# dump_path = '/home/em585489/mama_dataloader/testing'

################# Methods #######################################

def generate_envelope(bbox):
    env = [float('inf'), float('inf'), -float('inf'), -float('inf')]
    for bounding_box in bbox:
        x1, x2, y1, y2 = bounding_box
        env[0] = min(x1, env[0])
        env[1] = min(y1, env[1])
        env[2] = max(x2, env[2])
        env[3] = max(y2, env[3])

    return env

# Returning list of numpy arrays (the crops)
def generate_crops(clip, bbox, start, end, t_len, actors_involved):
    crops = []
    # list of numpy arrays 
    # print('this is the bbox: ', bbox)

    if bbox == []:
        print('THE BBOX IS EMPTY')
        return 

    bbox_idx = 0
    print('this is clip shape before cropping', clip.shape)
    if len(actors_involved) == 1:
        for f_id in range(start, end+1):
            frame = clip[f_id, :, :, :]
            b_box = bbox[bbox_idx]
            x1, y1, x2, y2 = b_box
            print('these are x1. y1. x2, y2',  x1, y1, x2, y2 )
            # print('frame before cropped frame', frame.shape)
            # print('y1:y2', y2-y1)
            print('this is frame before cropped_frame', frame.shape) 
            cropped_frame = frame[y1:y2, x1:x2, :]  
            if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                cropped_frame =  frame[x1:x2, y1:y2, :]  
            # cropped_frame = frame[x1:x2, y1:y2, :]  
            print('this is cropped_frame.shape', cropped_frame.shape)
            # exit()
            crops.append(cropped_frame)
            bbox_idx += 1
    else:
        print('going into the else part')
        for f_id in range(start, end-1):
            frame = clip[f_id, :, :, :]
            b_box = bbox[bbox_idx]
            x1, y1, x2, y2 = b_box
            print('these are x1. y1. x2, y2',  x1, y1, x2, y2 )
            # cropped_frame = frame[y1:y2, x1:x2, :] 
            print('this is frame before cropped_frame', frame.shape) 
            # cropped_frame = frame[x1:y1, x2:y2, :] 
            cropped_frame = frame[x2:y2, x1:y1, :]  
            print('this is cropped_frame.shape', cropped_frame.shape) 
            # exit()
            crops.append(cropped_frame)
            bbox_idx += 1

    # print('shape of numpy array', crops[0].shape)

    if crops == None or crops == []:
        for _ in range(end-start +1):
            crops.append(np.zeros((112, 112, 3)))
        # crops = [np.zeros((112, 112, 1) for _ in range(end-start + 1))]
    else:
        crops = [cv2.resize(img, (112, 112)) for img in crops]

    vid_len = len(crops)
    if vid_len >= t_len:
        rand_idx = random.randint(0, vid_len-t_len)
        crops = crops[rand_idx: rand_idx + t_len] 
    else:
        pad = t_len - vid_len
        for padding in range(pad):
            crops.append(np.zeros(crops[0].shape))


    return crops       

def extract_tubes(clip, vid_id, label_ids, dump_path=dump_path, tube_len=100):
    assert isinstance(label_ids, dict), 'The label_ids is not a dictionary!'
    # the error may print if the assert is written wrong

    #reading annotation 
    vid_dir = data_dir + '/' + vid_id
    ann_path = vid_dir + '/' + (vid_id + '.json')


    with open(str(ann_path),'r') as f:
        ann = json.load(f)

    # To put the info from the json into a dictionary 
    # the number of tubes should be the number of keys in the ann because thats the number of actions
    # tubes = [{} for _ in range(len(ann.keys()))]
    # tube_output is a list, len(ann.keys) because thats how many tubes there should be

    tubes = {}    # tubes and actions should be the same length because you are creating a tube for every action
    actions = []
    # for i in ann.keys(): actions.append(i)


    for action_id in ann.keys():     # was list(ann,keys())
        actions.append(action_id)
        actors_involved = ann[action_id]['actors']  # list of actors involved
        print('this is actors_involved', actors_involved)
        start = ann[action_id]['start_frame']
        end = ann[action_id]['end_frame']
        # print('start and end', start, end)
        # print('this is num of frames, should be same as num of detections', end-start +1)
        action_type = ann[action_id]['action_type']
        detections_dict = ann[action_id]['detections']

        actors_bbox = [] # it will be the bbox for each actor in each frame
        crop_bbox = []

        # dont make tubes from just 1 frame !!!!




        # crop_bbox should have list of coordinates for each frame (so len of crop_bbox should be end-start)
        # To check some parts of the json match up with the actual video clip
        # num_frames = video.shape[0]
        # start_end = end - start
        # if(num_frames < start_end):
        #     video = [np.array(range(start, end + 1))]

        # for frame in range(start, end+1):      # not sure if i need this line
        for frame_id in detections_dict.keys():
            for actor in actors_involved:
                actor = str(actor)
                if actor in detections_dict[frame_id].keys():
                    actors_bbox.append(detections_dict[frame_id][actor])
                else:
                    continue
            
            if not len(actors_involved) <= 1:
                temp_env = generate_envelope(actors_bbox)
                # 
                crop_bbox.append(temp_env)
                # print('this is the actors_bbox before clearing', actors_bbox)
                actors_bbox.clear()
            else:
                crop_bbox.append(actors_bbox[0])
        
        # print('this is the bbox, should be list of detections, len of it same as frames', len(crop_bbox))
        cropped_video = generate_crops(clip, crop_bbox, start, end, tube_len, actors_involved)
        # print('cropped_video[0]: ', cropped_video[0])

        if action_type in label_ids.keys():
            print('This is the action type', action_type)
            label_id = label_ids[action_type]
            label_id = str(label_id)
        else:
            print('ERROR, ACTION TYPE WAS NOT FOUND')

        # add folder to its directory
        if not os.path.exists(dump_path + '/' + label_id + '/' + vid_id):
            os.mkdir(dump_path + '/' + label_id + '/' + vid_id)
        video = cv2.VideoWriter(dump_path + '/' + label_id + '/' + vid_id + '/' + (vid_id + '.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (112, 112))  #112, 112
        for frame in cropped_video:
            video.write(np.uint8(frame))
        video.release()

        # THE VIDEO SHOULD BE IN THE CORRECT DIRECTORY CORRESPONDING TO ITS LABEL 

        # Making the tubes into tensors, then adding it properly to data structure w/ label
        # print('type of cropped_video', type(cropped_video))
        vid_numpy = np.stack(cropped_video, axis=0)
        tube_tensor = torch.from_numpy(vid_numpy)
        # print('this is tube_tensor shape', tube_tensor.shape)
        tube_tensor = torch.swapaxes(tube_tensor, 1, 3) # shape is now color_ch, H, W, num_frames
        # print('should be in num_frames x ch x w x h')
        # not sure if line above is needed, but the resnet model is usually color_ch X H X W X batch


        # IF THE LABEL IS ALREADY IN THERE YOU STILL NEED TO ADD IT , but tube tensor wont be the same for every vid right
        tubes[tube_tensor] = label_id
        # if not label_id in tubes.keys():
        #     tubes[label_id] = tube_tensor
        # else:
                

    # print('this is the actions list', actions)
    # print('this is the tubes dict', tubes)


    # for each action id, there is one tube so add a tube at the end of this for-loop 
    # have the label the key of the dict and the tensor tube as the value

    # SAVE IT TO THE TUBES DICTIONARY AND ALSO ADD THE LABELS TO IT AS WELL
    # VISUALIZE IT AND DUMP INTO THE RIGHT DIRECTORY 

    # STILL NEED TO WRITE CODE WITH THE ACTION TYPE 

    # tubes.append(THE CROP)
    # save the label with it USE THE ACTION TYPE


    if len(tubes) == len(actions):
        print('Tube extraction successful')
    else:
        print('Tubes seem to be off, there should be one tube for each action')
        print('len of actions', len(actions))
        print('len of tubes', len(tubes))

    return tubes

def dump_vid(save_path, v_id):
    # this method should see the label in the action type and put it in the approiate dumping directory 
    return 




if __name__ == '__main__':

    print('Making directories')
    # Only does it once in the beginning, not sure why it was all out of order
    if not os.path.exists('/home/em585489/mama_dataloader/tubes'):
        label_dir = [str(i+1 )for i in range(35)]
        for label_id in label_dir:
            path = os.path.join(dump_path, label_id)
            if not os.path.exists(path):
                os.mkdir(path)


    # print('Starting tube extraction')

    # ucf_mama = TinyActionsLoader('train', [224, 224], 1)
    # vid_id = ucf_mama.get_vid_id(0)
    # print('This is the video id: ', vid_id)
    # label_id = ucf_mama.cat_to_int # Dictionary for the action labels
    # print('Type of label_id, should be a dictionary ', type(label_id))
    
    # tube_ouput = extract_tubes(ucf_mama[0], vid_id=vid_id, label_ids=label_id, tube_len=ucf_mama.tube_len)

    #clip, vid_id, label_ids


    # video, bbox, label, _ , __ = ucf_mama.load_video() 
    # ucf_mama[0]
    # to do a bunch of the videos just do a for loop and go through all the indices of ucf_mama 
    # to get v_id >>> ucf_mama[0] == will give you that video at that index 



    # QUESTIONS TO ASK/PROBLEMS
    # how to get the original videos in order to crop/generate tubes, using the dataloader ?
    # because the ucf_mama[0] is not working (i sliced the self.vid_files in dataloader but not part of this problem, not sure how to use it)
    # Having problems with the original __getitem__ of the dataloader
    # Traceback (most recent call last):
        # File "/home/em585489/mama_dataloader/tubes_mama.py", line 257, in <module>
        #     tube_ouput = extract_tubes(ucf_mama[0], vid_id=vid_id, label_ids=label_id, tube_len=ucf_mama.tube_len)
        # File "/home/em585489/mama_dataloader/cvpr_dataset_loader.py", line 609, in __getitem__
        #     for activity_id in label[f_id]:
        # IndexError: list index out of range
    # 
    # MISC (for emily) if you want 
    # use from pathlib import Path instead of having to add strings together for the paths 

    


    


