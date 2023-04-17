import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys
import io
import argparse
import math
from efficientnet_pytorch import EfficientNet
import pkg_resources
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def softmax_transformation(scores):
    exp_scores = list(map(math.exp, scores))
    return [a/sum(exp_scores) for a in exp_scores]

def model_predict(model):
    preds_list = []
    indices_list = []
    scores_list = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (inputs, indices) in enumerate(testing_generator):
            inputs = inputs.to(device, dtype=torch.float)
            indices_list += indices.tolist()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            scores_list += [softmax_transformation(item) for item in outputs.tolist()]
            preds_list += preds.tolist()
        model.train(mode=was_training)
    return preds_list, indices_list, scores_list



class TestDataset(Dataset):
  def __init__(self, data_table):
        'Initialization'
        self.list_npys = data_table.npy_filepath.values

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_npys)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        npy_file = self.list_npys[index]
        # Load data and get label
        data = np.load(npy_file)
        X = torch.from_numpy(data.transpose(2,0,1)/255)
        return X, index


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i", "--input_file", required=True, help="the input feature file generated from the previous step.")
    parser.add_argument("-t", "--testing_mode", required=False, default=True, help="testing mode or training mode; True for testing mode.")
    parser.add_argument("-m", "--model", required=False, default="efficientnet-b4_epoch_6.pt", help="the convolutional neural network model \
                                                                                         transfer learning is based on.")
    parser.add_argument("-mp", "--model-path", required=False, help="if you want to use a model that you trained or modifed, you can input  \
                                                                                         the path to the model here. Make sure the matching \
                                                                                         model type is given in the -m argument")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=10, help="traing or testing batch size.")
    parser.add_argument("-o", "--output_file", required=True, help="prediction output file")
    parser.add_argument("-gb", "--build", required=True, help="genome build, use hg19 or hg38")
    options = parser.parse_args(args)
    return options

def check_x_region(positions):
    in_par1 = (positions >= x_par1_region[0]) & (positions <= x_par1_region[1])
    in_par2 = (positions >= x_par2_region[0]) & (positions <= x_par2_region[1])
    return (~in_par1) & (~in_par2)

def check_y_region(positions):
    in_par1 = (positions >= y_par1_region[0]) & (positions <= y_par1_region[1])
    in_par2 = (positions >= y_par2_region[0]) & (positions <= y_par2_region[1])
    return (~in_par1) & (~in_par2)

def prediction_decision(features_df, scores_list):
    predictions = np.array(["artifact"] * len(features_df), dtype = object)
    mosaic_scores = scores_list[:, -1].astype(float)
    depth_fractions = features_df.depth_fraction.astype(float)
    segdups = features_df.segdup.values.astype(int)
    all_repeats = features_df.all_repeat.astype(int)
    gnomads = features_df.gnomad.astype(object)
    gnomads[gnomads=="."] = 0
    gnomads = gnomads.astype(float)
    chroms = features_df.chrom.astype(str)
    positions = features_df.pos.astype(int)
    sexs = features_df.sex.astype(str)
    lower_CIs = features_df.lower_CI.astype(float)
    upper_CIs = features_df.upper_CI.astype(float)
    #mosaic
    mosaic_filters = (depth_fractions >= 0.6) & (depth_fractions <= 1.7) & (segdups == 0) & (all_repeats == 0) &\
              (gnomads < 0.001) & (mosaic_scores > 0.6)
    predictions[np.where(mosaic_filters)] = "mosaic"
    extra_mosaic_filters = (depth_fractions >= 0.6) & (depth_fractions <= 1.7) & (segdups == 0) & (all_repeats == 0) &\
              (gnomads < 0.001) & (lower_CIs >= 0.5) & (upper_CIs < 0.5)
    extra_mosaic_filters_X = extra_mosaic_filters & (sexs == "M") & (chroms == "X") & (check_x_region(positions))
    extra_mosaic_filters_Y = extra_mosaic_filters & (sexs == "M") & (chroms == "Y") & (check_y_region(positions))
    predictions[extra_mosaic_filters_X | extra_mosaic_filters_Y] = "mosaic"
    #heterozygous
    hetero_filters = (mosaic_scores <= 0.6) & (upper_CIs >= 0.5) & (lower_CIs < 0.5)
    predictions[np.where(hetero_filters)] = "heterozygous"
    #homozygous
    ref_homo_filters = (mosaic_scores <= 0.6) & (lower_CIs < 0.01) & (upper_CIs < 0.5)
    predictions[np.where(ref_homo_filters)] = "reference_homozygous"
    alt_homo_filters = (lower_CIs > 0.5) & (upper_CIs > 0.99)
    predictions[np.where(alt_homo_filters)] = "alternative_homozygous"
    return predictions.reshape(-1,1)
   

def main():
    options = getOptions(sys.argv[1:])
    if options.build == 'hg19':
        x_par1_region = [60001, 2699520]
        y_par1_region = [10001, 2649520]
        x_par2_region = [154931044, 155260560]
        y_par2_region = [59034050, 59363566]
    elif options.build == 'hg38':
        x_par1_region = [10001, 2781479]
        y_par1_region = [10001, 2781479]
        x_par2_region = [155701383, 156030895]
        y_par2_region = [56887903, 57217415]
    else:
        sys.stderr.write((options.build + " is an invalid genome build, please see help message")
        sys.exit(3)

    input_file = options.input_file
    mode = options.testing_mode
    model_name = options.model
    model_path = options.model_path
    batch_size = options.batch_size
    output_file = os.path.abspath(options.output_file)

    if not os.path.exists(input_file):
        sys.stderr.write("Please provide a valid input file.")
        sys.exit(2)

    output_dir = "/".join(output_file.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}


    model_type = model_name.split("_")[0]
    HERE = os.path.abspath(os.path.dirname(__file__))
    if not model_path:
        model_path = os.path.join(HERE, "models/" + model_name)

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

    sys.stdout.write("Loading input data...")
    features_df = pd.read_csv(input_file, sep="\t")
    features_header = features_df.columns
    sys.stdout.write("complete\n")
    global testing_generator
    testing_generator = DataLoader(TestDataset(features_df), **params)

    #make predcitions
    preds_list, indices_list, scores_list = model_predict(model)
    preds_list = np.array(preds_list).reshape(-1,1)
    features_df = features_df.loc[indices_list, :]
    scores_list = np.array(scores_list)
    #determine genotypes
    prediction_list = prediction_decision(features_df, scores_list)
    image_list = features_df.image_filepath.values.reshape(-1,1)
    header = ["#sample_name", "sex","chrom", "pos", "ref", "alt", "variant", "maf", "lower_CI", "upper_CI", "variant_type", "gene_id",
              "gnomad", "all_repeat", "segdup", "homopolymer", "dinucluotide", "depth_fraction",
              "score1", "score2", "score3", "prediction", "image_filepath"]
    results = np.hstack([features_df[features_header[:-2]].values, scores_list, prediction_list, image_list])
    results_pd = pd.DataFrame(results, columns = header)
    results_pd.to_csv(output_file, index=None, sep="\t")


