from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import os
import PIL
import torchvision.utils as vutils
import argparse
from sklearn.metrics import classification_report, accuracy_score
from torchvision import transforms

epsilon = 1e-7



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet50
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224


    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1, 2, 0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def evaluate_gt(root_image_dir, model_name, model_path):

    if args.dataset == 'pororo':
        from pororo_dataloader import ImageDataset, StoryImageDataset
    elif args.dataset == 'flintstones':
        from flintstones_dataloader import ImageDataset, StoryImageDataset
    else:
        raise ValueError

    # Number of classes in the dataset
    num_classes = 9
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    running_corrects = 0
    running_recalls = 0
    total_positives = 0

    phase = 'eval'
    is_inception = True if model_name == 'inception' else False

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode
    image_dataset = ImageDataset(root_image_dir, input_size, mode='val')
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 32

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    # Iterate over data.
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'train'):

            outputs = model_ft(inputs)
            if model_name == 'imgD':
                outputs = model_ft.cate_classify(outputs).squeeze()
            preds = torch.round(nn.functional.sigmoid(outputs))
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # statistics
        iter_corrects = torch.sum(preds == labels.float().data)
        xidxs, yidxs = torch.where(labels.data == 1)
        # print(xidxs, yidxs)
        # print([labels.data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)])
        iter_recalls = sum(
            [x.item() for x in
             [labels.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
        total_positives += xidxs.size(0)

        for l, p in zip(labels, preds):
            if torch.all(torch.eq(l.float().data, p)):
                image_accuracy += 1

        running_corrects += iter_corrects
        running_recalls += iter_recalls

    epoch_acc = running_corrects * 100 / (len(image_dataset) * num_classes)
    epoch_recall = running_recalls * 100 / total_positives
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, epoch_acc, epoch_recall))
    print('{} Story Exact Match Acc: {:.4f}%'.format(phase, float(story_accuracy) * 100 / len(image_dataset)))
    print('{} Image Exact Match Acc: {:.4f}%'.format(phase, float(image_accuracy) * 100 / len(image_dataset)))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    preds = np.round(1 / (1 + np.exp(-all_predictions)))
    print(classification_report(all_labels, preds, digits=4))

    # for i in range(0, 9):
    #     print("Character %s" % i)
    #     print(classification_report(all_labels[:, i], preds[:, i]))

    # Inception Score
    # all_predictions = all_predictions + epsilon
    # py = np.mean(all_predictions, axis=0)
    # print(py, py.shape)
    # split_scores = []
    # splits = 10
    # N = all_predictions.shape[0]
    # for k in range(splits):
    #     part = all_predictions[k * (N // splits): (k + 1) * (N // splits), :]
    #     py = np.mean(part, axis=0)
    #     scores = []
    #
    #     for i in range(part.shape[0]):
    #         pyx = part[i, :]
    #         scores.append(entropy(pyx, py))
    #     split_scores.append(np.exp(np.mean(scores)))
    # print("InceptionScore", np.mean(split_scores), np.std(split_scores))


def evaluate(args):

    root_image_dir, model_name, model_path = args.img_ref_dir, args.model_name, args.model_path

    if args.dataset == 'pororo':
        from pororo_dataloader import ImageDataset, StoryImageDataset
    elif args.dataset == 'flintstones':
        from flintstones_dataloader import ImageDataset, StoryImageDataset
    else:
        raise ValueError

    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    phase = 'eval'

    model_ft, input_size = initialize_model(model_name, args.num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_path))

    img_transform = transforms.Compose([
        # Image.fromarray,
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    # Create training and validation datasets
    try:
        image_dataset = StoryImageDataset(args.img_ref_dir, None, preprocess=img_transform,
                                          mode=args.mode,
                                          out_img_folder=args.img_gen_dir,
                                          return_labels=True)
    except TypeError:
        image_dataset = StoryImageDataset(args.img_ref_dir, None, transform=img_transform,
                                          mode=args.mode,
                                          out_img_folder=args.img_gen_dir,
                                          return_labels=True)
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 20

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    running_corrects = 0
    running_recalls = 0
    total_positives = 0

    # Iterate over data.
    no_char_images = 0
    for i, batch in tqdm(enumerate(dataloader)):

        inputs = batch[0]
        labels = batch[1]

        inputs = inputs.view(-1, n_channels, inputs.shape[-2], inputs.shape[-1])
        labels = labels.view(-1, labels.shape[-1])
        assert inputs.shape[0] == labels.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            outputs = model_ft(inputs)
            preds = torch.round(nn.functional.sigmoid(outputs))
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # statistics
        iter_corrects = torch.sum(preds == labels.float().data)
        xidxs, yidxs = torch.where(labels.data == 1)
        # print(xidxs, yidxs)
        # print([labels.data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)])
        iter_recalls = sum(
            [x.item() for x in [labels.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
        total_positives += xidxs.size(0)

        labels = labels.view(-1, labels.shape[-1])
        preds = preds.view(-1, labels.shape[-1])
        assert labels.shape[0] == preds.shape[0]


        for label, pred in zip(labels, preds):
            if not torch.any(label):
                no_char_images += 1
            if torch.all(torch.eq(label.float().data, pred)):
                image_accuracy += 1

        running_corrects += iter_corrects
        running_recalls += iter_recalls

    print("Frames with no images: ", no_char_images)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    # preds = np.round(1 / (1 + np.exp(-all_predictions)))
    print(classification_report(all_labels, all_predictions, digits=4))
    print("Accuracy: ", accuracy_score(all_labels, all_predictions))

    epoch_acc = float(running_corrects) * 100 / (all_labels.shape[0] * all_labels.shape[1])
    epoch_recall = float(running_recalls) * 100 / total_positives
    print('Manually calculated accuracy: ', epoch_acc)
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, all_predictions), epoch_recall))
    print('{} Image Exact Match (Frame) Acc: {:.4f}%'.format(phase, image_accuracy * 100 / all_labels.shape[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate for Character Recall & InceptionScore')
    parser.add_argument('--dataset',  type=str, default='pororo')
    parser.add_argument('--img_ref_dir',  type=str, required=True)
    parser.add_argument('--img_gen_dir',  type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ground_truth', action='store_true')
    args = parser.parse_args()

    if args.ground_truth:
        evaluate_gt(args)
    else:
        evaluate(args)

    # numpy_to_img(os.path.join(args.image_dir, 'images-epoch-%s.npy' % args.epoch_start),
    #              os.path.join(args.image_dir, 'images-epoch-%s/' % args.epoch_start), 299)

