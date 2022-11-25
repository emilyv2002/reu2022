import os
import sys
import math
import time
from functools import partial
from resnet3D import ResNet, generate_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchvision import datasets, transforms, models
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torch import batch_norm_elemt, optim
import time
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
# from pytorch_i3d import InceptionI3d
import numpy as np
# from torch import optim
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# from torch.utils.data import *
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR
from dataloader_testing import TinyActionsLoaderTesting
# from dataloader_training import TinyActionsLoader

################### Initalize Parameters #####################################

print('==> Initializing Parameters....\n')

data_path = glob.glob('/home/em585489/mama_dataloader/tubes/**/*.avi', recursive=True)

# USE_CUDA = True if torch.cuda.is_available() else False
USE_CUDA = True if torch.cuda.is_available() else False
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 12
VAL_BATCH_SIZE = 3
NUM_EPOCHS = 150  #used to be 100
LEARNING_RATE = 0.001   #this is the initial learning rate
LR_SCHEDULER = True
collate_batch = 1
LR_step_size = 60
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

load_prev_weights = True # don't have any previous weights
resume = False
seed = 0
model_name = 'resnet3D'

########### Data ##################################################################

print('\n==> Preparing Data...')

#     # how to do padding with each batch size so they can be of equal length 
#     for tube in data:
#          add padding because some of the tubes were less than tube_len so they all need to be equal 
#      for tube, label in data:
#           if tube.shape[0] != 100 

# mean, std_dev = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def collate_fn(batch):

    batch_output = []
    print('len of batch', len(batch))

    for input in batch:
        result = []
        video, label = input[0], input[1]
        if video.shape[3] == 150:
            video = video[:, :, :, :100]
        result.append(video)
        result.append(label)

        batch_output.append(result)



    # for video, label in input:
    #     print('this is type of input', type(input))
    #     if video.shape[3] == 150:
    #         video = video[:, :, :, :100]
    #     result.append(video)
    #     result.append(label)
    # batch_output.append(result)

    # for video, label in batch:
    #     print('this is the batch, ', batch)
    #     print('this is type of batch[0]', type(batch[0]))
    #     print('len of batch', len(batch))
    #     # label_list.append(label)
    #     if video.shape[3] == 150:
    #         video = video[:, :, :, :100]
        # input_vid.append(video)

    # input_vid = pad_sequence(input_vid, batch_first=True, padding_value=0)

    return batch_output



# Train Data
# transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(), transforms.Normalize(mean, std_dev)])
# training_set = TinyActionsLoader('train', [112, 112], TRAIN_BATCH_SIZE)
# # # clip_size may be [224, 224], not sure 
# train_loader = DataLoader(
#         dataset=training_set,
#         batch_size=TRAIN_BATCH_SIZE,
#         num_workers=0,
#         collate_fn=None,
#         shuffle=True
#         )

testing_set = TinyActionsLoaderTesting('test', [112, 112], TRAIN_BATCH_SIZE)
test_loader = DataLoader(
    dataset=testing_set,
    batch_size=TEST_BATCH_SIZE,
    num_workers=0,
    collate_fn=None,
    shuffle=False
)


# Dataloader(training_set, TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
#not sure if 2 is enough for num_workers, may be a little more? 

# Validation Data
# validation_set = TinyActionsLoader('validation',[112, 112], VAL_BATCH_SIZE)  # dont think TinyActionsLoader has a validation set 
# val_data_loader = DataLoader(
#             dataset=validation_set,
#             batch_size=VAL_BATCH_SIZE,
#             num_workers=2,
#             shuffle=False
#         )

#num_workers is 2...... 

############ Model ###########################################################################

# model = torch.load('/home/em585489/mama_dataloader/trained_weights/10_weight/ucf_mama_10.pth')
path = '/home/em585489/mama_dataloader/trained_weights/100_weight/ucf_mama_100.pth'
# path = '/home/em585489/mama_dataloader/pretrained_weights/data/r3d18_K_200ep.pth'
# model = torch.load(path)
model = generate_model(18)
model.load_state_dict(torch.load(path))

# output = model.forward(training_set)
# model = ResNet()

########### Optimizer & Loss #####################################################################


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0)

if USE_CUDA:
    model = model.to(device)
    # criterion = criterion.to(device)

# optim.SGD(model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.999], weight_decay=0, eps=1e-6)
# should i used SGD, read somewhere Adam was better for ResNet
# momentum=0.9 ? also, do i need the betas 
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, step_size=LR_step_size, gamma=0.1)
scheduler = StepLR(optimizer, step_size=LR_step_size, gamma=0.1)

########## Training ##############################################################################
train_accuracy = []
train_losses = []

def train_one_epoch(epoch):
    start_time = time.time()
    model.train(mode=True)
    # steps = len(train_loader)
    # train_loss, test_loss = [], []     # don't need test_loss because we aren't testing in this
    # correct = 0
    # total = 0
    running_loss = 0
    correct = 0
    total = 0
    
    # should make them lists tho because of the videos?

    print(' ENTERING THE FOR LOOP IN TRAIN ONE EPOCH')
    for i, data in enumerate(train_loader):
        
        
        # print('BEFORE THE INPUTS AND LABELS')
        # print('len of data and type', len(data), type(data))
        inputs, labels = data
        # if inputs.shape[4] == 150:
        #     inputs = inputs[:, :, :, :, :100]
        # inputs = inputs.float()
        # labels = torch.Tensor(labels)
        # print('this is the type of input and labels', type(inputs), type(labels), inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()
        # labels = labels.float()
        # inputs = inputs.type(torch.LongTensor)
        # labels = labels.type(torch.LongTensor)
        outputs = model(inputs)
        #     # device = 
        #     # .to(device)
        
        # inputs = inputs.cuda()
        # model.forward()

        # print('GOT PASS INPUTS AND LABELS, BEFORE OPTIMIZER')
        # Clear gradients
        optimizer.zero_grad()
        # outputs = model(inputs)
        # print('GOT PAST THE OUTPUTS')
        loss = criterion(outputs, labels)
        loss.backward()
        # print('GOT PAST LOSS')
        # Update Weights
        optimizer.step()
        running_loss += loss.item()
        _ , predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print('GOT PAST OPTIMIZER AND RUNNING LOSS')
        # running_loss += loss.data[0]

        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch * len(train_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

        # train_loss += loss.item()
        # _, predicted = outputs.max(dim=1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     # inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = model(inputs)

    #     loss = criterion(outputs, targets)
    #     loss.backward()

    #     optimizer.step()

    #     train_loss += loss.item()
    #     _, predicted = outputs.max(dim=1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets).sum().item()


    time_elapsed = time.time() - start_time
    print('Training time: ', time_elapsed)
    hrs, _min, sec = time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60  # might not use

    train_loss = running_loss/len(train_loader)
    accu = 100.*correct/total
    train_accuracy.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

    



    # print('Epoch: %d | Time: %02d:%02d:%02d' % (epoch, hrs, _min, sec, running_loss/len(train_loader)))
    # | Loss: %.3f | Acc: %.2f%% [Train]'
    #    % (epoch, hrs, _min, sec, train_loss/len(train_loader), 100.*correct/total))


######## Validation ###############################################################################

# this validate method is for when the model and val_data_loader is initialized within the for-loop
# def validate(epoch, train_loss, model='Resnet'):
#     steps = len(val_data_loader)
#     # print('validation: batch size ', VAL_BATCH_SIZE, ' ', N_EPOCHS, 'epochs', steps, ' steps ')

#     min_valid_loss = np.inf


#     for e in range(NUM_EPOCHS):
#         val_loss = 0
#         model.eval() 
#         for data, labels in val_data_loader:
#             if torch.cuda.is_available():
#                 data, labels = data.cuda(), labels.cuda()
            
#             target = model(data)
#             loss = criterion(target, labels) # not too sure about this
#             valid_loss = loss.item() * data.size(0)

#         print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_data_loader)}')
#         if min_valid_loss > valid_loss:
#             print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
#             min_valid_loss = valid_loss
#             # Saving State Dict
#             torch.save(model.state_dict(), 'saved_model.pth')


    
    
####### Testing #####################################

# overall accuracy, and classwise accuracy 
# overall_accu = metric.accuracy_score()

eval_losses = []
eval_accu = []


labels_testing = testing_set.cat_to_int
correct_pred = {label_id: 0 for label_id in labels_testing.values()}
total_pred = {label_id: 0 for label_id in labels_testing.values()}

def test_one_epoch(epoch):
    start_time = time.time()
    model.eval()

    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            video, label = data
            video, label = video.to(device), label.to(device)
            video = video.float()
            # label = label.long()

            # print('this is the video', video.shape)
            # print('this is the label', label)

            outputs = model(video)
            # label = label.long() label.view(len(label), 1)
            loss = criterion(outputs, label)


            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()


    test_loss = running_loss/len(test_loader)
    accur = 100.*(correct/total)
    # eval_losses.append(test_loss)
    eval_accu.append(accur)
    time_elapsed = time.time() - start_time
    hrs, _min, sec = time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60

    # print('Time: %02d:%02d:%02d' % hrs, _min, sec)
    print('Time:', time_elapsed)
    # print('Epoch: ', epoch, 'Testing Accuracy: ', accur)

    # with torch.no_grad():
    #     for data in test_loader:
    #         video, labels = data
    #         video, labels = video.to(device), labels.to(device)
    #         video = video.float()
            
    #         outputs = model(video)
    #         _, predictions = torch.max(outputs, 1)

    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[label] += 1


    #             label = int(label)
    #             total_pred[label] += 1

    # for label_id, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[label_id]
    #     print(f'Accuracy for class: {label_id} is {accuracy:.1f}%')



    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accur))


    with torch.no_grad():
        for video, labels in test_loader:
            # video, labels = data
            video, labels = video.to(device), labels.to(device)
            video = video.float()
            
            outputs = model(video)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                label = label.tolist()
                # print('this is the label', label)
                if label == prediction:
                    correct_pred[label] += 1

                total_pred[label] += 1

    for label_id, correct_count in correct_pred.items():
        if correct_count == 0:
            continue
        accuracy = 100 * float(correct_count) / total_pred[label_id]
        print(f'Accuracy for class: {label_id} is {accuracy:.1f}%')







# DONT FORGET:
# add linear layers to the top of the model !!!!
# WRITE CODE FOR SAVED WEIGHTS 


if __name__ == '__main__':

    print('TRAINING BATCH SIZE: ', TRAIN_BATCH_SIZE)
    print("TEST_BATCH_SIZE: ", TEST_BATCH_SIZE)
    print("NUM OF EPOCHS: ", NUM_EPOCHS)
    print("Learning Rate: ", LEARNING_RATE)
    print('Resume Training:', 'Yes' if resume else 'No\n')
    # print("Masking mode: ", IS_MASKING)
    # print("pretrained_load: ", pretrained_load)
    # print("load previous weights: ", load_previous_weights)
    # print("Hybrid mode: ", HYBRID)
    # print("Percent: ", percent)

    # if USE_CUDA:
    #     model = model.cuda()


    # should i initialize training and val dataloader for every epoch ? 
    # print('==> Training model....\n')
    print('==> Testing model....\n')
    # print('lr = %g\n' % LEARNING_RATE)

    # tb_writer = SummaryWriter('/home/em585489/mama_dataloader')

    for epoch in range(1, NUM_EPOCHS+1):
        print('EPOCH:', epoch)

        # train(model, train_loader, optimizer, criterion, epoch)

        # train_one_epoch(epoch, tb_writer, train_loader)

        # train_one_epoch(epoch)
        test_one_epoch(epoch)

        # validate(epoch=epoch)

        # TO update the learning rate according to the scheduler
        scheduler.step()

        # path = '/home/em585489/mama_dataloader/trained_weights' + '/' + str(epoch) + '_weight'

        # if not os.path.exists(path):
        #     os.mkdir(path)
        

        # path = os.path.join(path, 'ucf_mama_' + str(epoch) + '.pth') 
        # torch.save(model.state_dict(), path)


    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. Epoch')
    plt.show()

    plt.plot(eval_accu, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.show()




    


    
   
    