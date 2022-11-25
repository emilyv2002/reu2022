# just forward pass (with trained_weights) to get predicted, convert to list

import numpy as np
import torch
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from resnet3D import ResNet, generate_model


def eval_training(predictions, model):
    # not sure if i need the model parameter
    n_classes = 35
    pred = [1, 3, 2, 1, 4]
    ground_truth = [1, 1, 4, 1, 4]
    class_id = 1
    idx = np.where(np.array(ground_truth) == class_id)[0].tolist()
    pred = np.array(pred)[idx]
    n_correct = np.sum(pred == 1)
    n_total = len(idx)
    acc = (n_correct/n_total)*100
    print('This is the accuracy for 1 Read_Document: ', acc)


    for i in range(n_classes):
        class_id = i+1
        pred = torch.argmax(predictions, axis=1)

        ground_truth = label





def eval_training(predictions, model):
    # These are the predicted classes 
    n_classes = 35
    # pred = (batch_size, 35)

    pred = torch.argmax(predictions, axis=1) # should be a list 
    return




if __name__ == '__main__':
    # This is for plotting the accuracy and loss over 100 epochs for training 


    ### Ploting Epoch vs Accuracy ####
    plt.plot()



    ### Plotting Epoch vs Loss ###


