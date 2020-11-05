import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys
import argparse
import pickle
import math
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VariantDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_table):
        'Initialization'
        self.labels = data_table[:,1]
        self.list_npys = data_table[:,0]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        npy_file = self.list_npys[index]
        # Load data and get label
        X = torch.from_numpy(np.load(npy_file).transpose(2,0,1)/255)
        y = int(self.labels[index])
        return X, y

def model_train(model, criterion, optimizer, scheduler, num_epochs, model_type):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_losses = []
    for epoch in range(num_epochs):
        sys.stdout.write('Epoch {}/{}'.format(epoch, num_epochs - 1)+"\n")
        sys.stdout.write('-' * 10 + "\n")
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            all_running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            i = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == "train":
                        if model_type.startswith("inception"):
                            outputs, aux_outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss =  criterion(outputs, labels) + criterion(aux_outputs, labels)
                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        if i % 100 == 99:
                            running_losses.append(running_loss/100)
                            running_loss = 0.0
                        i += 1
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    # statistics
                    all_running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc) + "\n")
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    sys.stdout.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    sys.stdout.write('Best val Acc: {:4f}'.format(best_acc) + "\n")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, running_losses


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i", "--input_file", required=True, help="the labeled input file; required field: npy_filepath, label")
    parser.add_argument("-e", "--train_epoch", required=False, type=int, default=0, help="conduct training on user-provided labeled \
                                                                                        sample from your own data set with \
                                                                                        provided number of epochs to train.")
    parser.add_argument("-m", "--model", required=False, default="efficientnet-b0", help="the convolutional neural network model \
                                                                                         transfer learning is based on.")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=10, help="traing or testing batch size.")
    parser.add_argument("-o", "--output_file", required=True, help="prediction output file")
    options = parser.parse_args(args)
    return options


def main(argv):
    options = getOptions(sys.argv[1:])
    input_file = options.input_file
    epoch = opions.train_epoch
    model_name = options.model
    batch_size = options.batch_size
    output_file = os.path.abspath(options.output_file)

    if not os.path.exists(input_file):
        sys.stderr.write("Please provide a valid input file.")
        sys.exit(2)

    #user provided input data has two columns: npy_filepath, label (note: npy_filepath could be obtained from deepmosaic-draw)
    data = pd.read_csv(input_file, sep="\t")

    output_dir = "/".join(output_file.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}

    model_type = model_name.split("_")[0]
    model_path = pkg_resources.resource_filename('deepmosaic', 'models/' + model_name)

    #model_name = os.path.abspath(model_path).split("/")[-1]
    if model_name.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(model_type)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(model_path,map_location=device))
        model = model.to(device)

    elif model_name.startswith("densenet"):
        model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(model_path,map_location=device))
        model = model.to(device)

    elif model_name.startswith("inception"):
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(model_path,map_location=device))
        model = model.to(device)

    elif model_name.startswith("resnet"):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(model_path,map_location=device))
        model = model.to(device)


    #start training-----------------------------------------
    for i in range(epoch):
        train_data = data[:int(len(data)*0.8), :]
        validation_data = data[int(len(data)*0.8):, :]

        training_generator = DataLoader(VariantDataset(train_data), **params)
        validation_generator = DataLoader(VariantDataset(validation_data), **params)

        global dataloaders
        global dataset_sizes

        dataloaders= {"train": training_generator, "val": validation_generator}
        dataset_sizes = {'train':len(train_data), "val": len(validation_data)}
        #initialize criterion etc.
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        #pseudo training
        model, loss_list = model_train(model, criterion, optimizer_ft, exp_lr_scheduler, 1, model_name)
        #save your model
        torch.save(model.state_dict(), output_file)
        np.save(output_dir + "/training_loss.npy", np.array(loss_list))
        sys.stdout.write("complete\n")


