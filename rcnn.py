# Import necessary libraries
import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import pdb  # Python Debugger
from sklearn.model_selection import train_test_split
from torchvision import datasets, models
import multiprocessing
import torchvision
import warnings
from PIL import Image
import numpy as np
import wandb  # Weights & Biases for tracking experiments
from tqdm import tqdm  # Progress bar
import torchvision.transforms.v2 as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate  # Custom training and evaluation functions
from torchvision.transforms.v2 import functional as F
from torch.utils.data.dataloader import default_collate
import utils  # Utility functions

# Define a custom class for extending the functionality of a pre-trained model using the RCNN (Region-based Convolutional Neural Network) architecture.
class CustomRCNN(nn.Module):
    """
    A custom implementation of the RCNN model for specific adjustments such as adding custom output activation layers.
    This class is built upon a given pre-trained model and modifies its output to include a log softmax layer.

    Attributes:
        model (torch.nn.Module): The original pre-trained model which will be used as the backbone for feature extraction.
        output_act (torch.nn.Module): The output activation layer.
    """

    # Initialize the CustomRCNN class with an original pre-trained model and the number of classes for classification.
    def __init__(self, original_model, num_classes):
        """
        Initialize the CustomRCNN with a pre-trained model and set up any additional layers or transformations that will be applied to the model's output.

        Parameters:
        original_model (torch.nn.Module): The pre-trained model.
        num_classes (int): The number of classes in the dataset.
        """
        super(CustomRCNN, self).__init__()  # Initialize the superclass (nn.Module)
        self.model = original_model  # Store the original model
        self.output_act = nn.LogSoftmax(dim=-1)  # Define the output activation function with log softmax applied across the last dimension

    # Define the forward pass through the network.
    def forward(self, x):
        """
        Define the forward pass procedure for an input batch.

        Parameters:
        x (Tensor): The input tensor containing batched image data.

        Returns:
        Tensor: The output tensor after passing through the model and the output activation function.
        """
        out = self.model(x)  # Pass the input through the original model to obtain features
        out = self.output_act(out)  # Apply the log softmax activation function to the features
        return out  # Return the activated output



# Custom dataset class for managing road sign image data
class Road_Dataset(Dataset):
    # Constructor for the Road_Dataset class
    def __init__(self, img_dir, df, transform=None):
        """
        Initialize the dataset with the directory of images, a DataFrame containing the metadata,
        and a transform to be applied to each image.

        Parameters:
        img_dir (str): Directory where the images are stored.
        df (pandas.DataFrame): DataFrame containing image metadata, including file paths and class IDs.
        transform (callable, optional): The function/transform that takes in an image and returns a transformed version.
        """
        self.img_dir = img_dir  # Store the directory path of images
        self.transform = transform  # Store the transformation to be applied to images
        self.train_df = df  # Store the DataFrame containing paths and annotations
        self.file_name = self.train_df['Path']  # Extract the column that contains image file names
        self.num_classes = len(set(self.train_df['ClassId']))  # Compute the number of unique classes

    # Return the total number of images
    def __len__(self):
        """
        Return the total number of items in the dataset.
        
        Returns:
        int: Total number of images
        """
        return len(self.train_df)  # The length is simply the number of rows in the DataFrame

    # Retrieve an image and its corresponding label at the specified index
    def __getitem__(self, idx):
        """
        Fetch the image and its corresponding label at the specified index in the dataset.

        Parameters:
        idx (int): Index of the image to retrieve.

        Returns:
        tuple: A tuple containing the transformed image and its bounding box coordinates.
        """
        # Construct the path to the image file
        img_path = os.path.join(self.img_dir, self.file_name.iloc[idx])
        # Load the image from the filesystem
        img = read_image(img_path)
        # Convert the loaded image to a PyTorch tensor
        img = tv_tensors.Image(img)

        # Retrieve bounding box coordinates stored in the DataFrame
        x1 = self.train_df.loc[self.train_df['Path'] == self.file_name.iloc[idx], 'Roi.X1'].values[0]
        y1 = self.train_df.loc[self.train_df['Path'] == self.file_name.iloc[idx], 'Roi.Y1'].values[0]
        x2 = self.train_df.loc[self.train_df['Path'] == self.file_name.iloc[idx], 'Roi.X2'].values[0]
        y2 = self.train_df.loc[self.train_df['Path'] == self.file_name.iloc[idx], 'Roi.Y2'].values[0]

        # Apply any transformations to the image, if specified
        if self.transform:
            img = self.transform(img)

        # Return the transformed image along with the bounding box coordinates as a dictionary
        return {'image': img, 'bbox': [x1, y1, x2, y2]}


def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader.
    This function prepares a batch of data by separately handling images and their corresponding annotations.

    Parameters:
    batch (list of dict): A list of dictionaries, where each dictionary contains information about a single data item,
                          including fields like 'image' for the image tensor, and 'bbox' for bounding box coordinates.

    Returns:
    dict: A dictionary with the same keys as the input items but where the data is now batched. 
    """
    # Unpack the images and bounding boxes from the batch
    images = [item['image'] for item in batch]  # Extract all images
    bboxes = [item['bbox'] for item in batch]  # Extract all bounding boxes

    # Stack all images into a single tensor to create a batch
    images = torch.stack(images, dim=0)

    # Return the batched data as a dictionary
    return {'image': images, 'bbox': bboxes}


def accuracy(outputs, labels):
    """
    Calculate the accuracy of predictions.

    Parameters:
    outputs (Tensor): The predictions made by the model.
    labels (Tensor): The ground truth labels corresponding to the inputs that generated the outputs.

    Returns:
    float: The accuracy of the predictions as a percentage, where 100% represents perfectly accurate predictions.
    """
    # Convert outputs to predicted class indices if outputs are not already class indices (e.g., if outputs are probabilities)
    predicted = outputs.argmax(dim=1, keepdim=True)  # Assuming outputs are logits or probabilities

    # Calculate the number of correctly predicted items
    correct = predicted.eq(labels.view_as(predicted)).sum().item()  # Count how many predictions match the labels

    # Calculate total number of items
    total = labels.size(0)

    # Compute accuracy
    accuracy = 100.0 * correct / total  # Convert correct count to percentage of the total items

    return accuracy


def evaluate_predictions(pred_dicts, gt_dicts, iou_threshold=0.5):
    """
    Evaluate model predictions against ground truth annotations using metrics such as precision, recall, and average IoU.

    Parameters:
    pred_dicts (list of dicts): List containing dictionaries of predictions for each image in the batch. Each dictionary
                                includes 'boxes' (predicted bounding boxes), 'labels' (predicted labels), and 'scores' (confidence scores).
    gt_dicts (list of dicts): List containing dictionaries of ground truths for each image. Each dictionary has 'boxes' (true bounding boxes)
                              and 'labels' (true labels).
    iou_threshold (float): Threshold value to determine when a prediction is considered correct based on IoU metric.

    Returns:
    float, float, float: Precision, recall, and average IoU across the evaluated batch.
    """

    # Initialize counters for true positives, false positives, and false negatives
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    iou_sum = 0  # Sum of IoU values for calculating average IoU
    iou_count = 0  # Counter for number of IoUs to calculate average

    # Iterate over each set of predictions and corresponding ground truth
    for pred_dict, gt_dict in zip(pred_dicts, gt_dicts):
        pred_boxes = pred_dict['boxes']  # Predicted bounding boxes
        pred_labels = pred_dict['labels']  # Predicted labels of the bounding boxes
        pred_scores = pred_dict['scores']  # Confidence scores of the predictions

        gt_boxes = gt_dict['boxes']  # Ground truth bounding boxes
        gt_labels = gt_dict['labels']  # Ground truth labels

        # Validate that each ground truth dict contains exactly one box and label, as expected
        if len(gt_boxes) != 1:
            raise ValueError("Each gt_dict must contain exactly one ground truth box and label.")

        # Handle cases with no predictions
        if len(pred_boxes) == 0:
            total_false_negatives += 1
            continue

        # Sort predictions by confidence scores in descending order to prioritize high confidence predictions
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        # Track whether the ground truth has been matched
        matched_gt = False

        # Evaluate each predicted box against the ground truth box
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)  # Compute IoU between predicted and gt box
            best_iou = ious.squeeze(0).item()  # Extract the IoU value

            # Check if the prediction matches the ground truth criteria
            if best_iou >= iou_threshold and not matched_gt and gt_labels[0] == pred_label:
                total_true_positives += 1
                matched_gt = True
                iou_sum += best_iou
                iou_count += 1
            else:
                total_false_positives += 1

        # If no prediction matched the ground truth, count it as a false negative
        if not matched_gt:
            total_false_negatives += 1

    # Calculate precision, recall, and average IoU
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    average_iou = iou_sum / iou_count if iou_count > 0 else 0

    return precision, recall, average_iou


def finetune():
    # Check if wandb (Weights & Biases) logging is enabled for experiment tracking
    if wandb_on:
        # Initialize a new wandb run with a specific group name for organization
        wandb.init(group='frcnn')
        # Access the wandb configuration set during the sweep or previous setup
        config = wandb.config
        # Set batch size, learning rate, and regularization strength from wandb configuration
        batch_size = config.batch
        learning_rate = config.learning_rate
        reg = config.reg
        # Set patience for early stopping to 5 epochs
        patients = 5
    else:
        # Default settings if wandb is not enabled
        learning_rate = 1e-5  # Default learning rate
        patients = 5  # Default patience for early stopping
        reg = 0.001  # Default regularization strength
        batch_size = 32  # Default batch size

    # Suppress specific warnings from libpng
    warnings.filterwarnings("ignore", message="libpng warning")

    # Load the dataset from a CSV file
    raw_csv = pd.read_csv('gtsrb-german-traffic-sign/Train.csv')
    # Filter the dataset to use only the first 100 entries for each class to ensure dataset balance
    filtered_df = raw_csv.groupby('ClassId').head(100)

    # Split the filtered data into training and validation sets with a stratified split to maintain class balance
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, stratify=filtered_df['ClassId'])

    # Define the target image width for resizing operations
    target_width = 225
    # Define the transformations to be applied to training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            # Resize images to slightly larger than target due to potential size variations and cropping needs
            transforms.Resize(target_width + 1, antialias=True),
            # Crop images to the desired size
            transforms.CenterCrop(target_width),
            # Apply Gaussian blur for data augmentation
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            # Convert image data to floating point tensor and scale pixel values
            transforms.ToDtype(torch.float, scale=True),
            # Convert image data into a pure tensor format
            transforms.ToPureTensor(),
            # Normalize image data using pre-defined mean and standard deviations for model compatibility
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # Similar transformations for validation data, but without augmentation like Gaussian blur
            transforms.Resize(target_width + 1, antialias=True),
            transforms.CenterCrop(target_width),
            transforms.ToDtype(torch.float, scale=True),
            transforms.ToPureTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create instances of the Road_Dataset for training and validation. These datasets will apply specified transformations
    # to images located in the given directory and use metadata from the respective dataframes.
    train_dataset = Road_Dataset('/local/scratch/a/ko120/FRCNN/gtsrb-german-traffic-sign',
                                train_df,
                                data_transforms['train'])

    val_dataset = Road_Dataset('/local/scratch/a/ko120/FRCNN/gtsrb-german-traffic-sign',
                            val_df,
                            data_transforms['val'])

    # Set the number of worker processes for loading data.
    num_cores = 0

    # Setup the data loaders for the training and validation datasets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=num_cores, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_cores, collate_fn=collate_fn)

    # Extract the number of classes from the training dataset, which is used to configure the model's final layer appropriately.
    num_classes = train_dataset.num_classes

    # Initialize the model using a pre-trained Faster R-CNN with a MobileNet V3 backbone. The pre-trained weights are from the COCO dataset
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='COCO_V1')

    # Get the number of input features for the classifier used in the ROI (Region of Interest) heads from the model.
    # This is necessary to modify the head for the specific number of output classes in the current dataset.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head of the model with a new one suited for the actual number of classes in the dataset.
    # FastRCNNPredictor is used to create a new predictor layer with the correct number of inputs and outputs.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)

    # Set the number of epochs for training, which defines how many times the model will see the entire dataset.
    num_epochs = 150

    # Define an interval for printing out the training progress and statistics to monitor the training process.
    print_interval = 10

    # Initialize the best validation accuracy to zero to keep track of the highest accuracy achieved during validation
    best_val_acc = 0

    # Configure the optimizer, using Stochastic Gradient Descent with momentum and weight decay for regularization.
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=learning_rate, weight_decay=reg, momentum=0.9)

    # Set the loss criterion to Negative Log Likelihood Loss, suitable for classification problems with log probabilities as input
    criterion = nn.NLLLoss()

    # Learning rate scheduler to adjust the learning rate at predefined milestones during training
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * num_epochs), int(0.9 * num_epochs)]
    )

    # Training and validation loop
    for epoch in range(num_epochs+1):
        # Variables to accumulate metrics for analysis
        total_loss = 0
        total_correct = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        precisions = []
        recalls = []
        avg_ious = []

        # Set the model to training mode (enables dropout, batch normalization)
        model.train()
        for data in train_loader:
            images = data[0].to(device)  # Move images to the appropriate device (GPU or CPU)
            targets = data[1]  # Targets contain labels and possibly other annotations like bounding boxes
            # Convert targets to appropriate device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()  # Clear gradients before calculating new ones
            loss_dict = model(images, targets)  # Forward pass and loss calculation
            losses = sum(loss for loss in loss_dict.values())  # Aggregate losses for backpropagation
            losses.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            scheduler.step()  # Step the scheduler
            
            train_losses.append(losses.detach())  # Store losses for later analysis

        # Calculate average training loss over the epoch
        epoch_train_loss = sum(train_losses) / len(train_losses)

        # Log training results periodically
        if epoch % print_interval == 0:
            print("Epoch {}, Train_Loss={}".format(epoch, round(epoch_train_loss.item(), 4)))

        # Validation phase
        model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization)
        with torch.no_grad():  # Disable gradient computation during evaluation
            for data in val_loader:
                images = data[0].to(device)
                targets = data[1]
                targets = [{k: v.to(device) for k, v in t.items()}}

                # Evaluate the model's predictions
                pred_dict = model(images, targets)
                precis, recall, avg_iou = evaluate_predictions(pred_dict, targets)
                precisions.append(precis)
                recalls.append(recall)
                avg_ious.append(avg_iou)
                
        # Calculate validation metrics
        epoch_val_prec = sum(precisions) / len(precisions)
        epoch_val_recal = sum(recalls) / len(recalls)
        epoch_val_iou = sum(avg_ious) / len(avg_ious)

        # Check if the current epoch's validation precision is the best and update accordingly
        if best_val_acc < epoch_val_prec:
            best_val_acc = epoch_val_prec  # Update best validation accuracy
            if wandb_on:  # Log this to Weights & Biases if enabled
                wandb.log({'best_val_acc': best_val_acc})
            torch.save(model.state_dict(), 'model_best_val_acc.pth')  # Save the model with the best accuracy
            early_stop_count = 0  # Reset early stopping counter
        else:
            early_stop_count += 1  # Increment the early stopping counter if no improvement

        # Stop training if no improvement in validation accuracy for a number of epochs equal to 'patients'
        if early_stop_count > patients:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

        # Log validation results periodically
        if epoch % print_interval == 0:
            print("Epoch {}, Val_precision={}, Val_recall={}, Val_iou={}".format(epoch, round(epoch_val_prec, 4), round(epoch_val_recal, 4), round(epoch_val_iou, 4)))


        #wandb loging
        if wandb_on:
            wandb.log({'epoch':epoch, 'train_loss':epoch_train_loss, 'val_precision':epoch_val_prec, 'val_recal':epoch_val_recal, 'val_iou':epoch_val_iou})
       
    return


# This block is used to ensure the code only runs when the script is executed directly (not imported as a module).
if __name__ == '__main__':
    # preprocess_img('train')

    # Determine if CUDA is available in the system, and set the device to GPU if available, otherwise use CPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # A flag to control whether to use wandb for tracking experiments and hyperparameter tuning.
    wandb_on = True

    # Check if wandb logging is enabled
    if wandb_on:
        # Configuration for a wandb sweep, used for systematic hyperparameter tuning.
        sweep_config = dict()
        # Define the sweep method as 'grid' which will explore all combinations of hyperparameters.
        sweep_config['method'] = 'grid'
        # Set the objective or metric to maximize during the sweep; here it is validation accuracy.
        sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
        # Define the hyperparameters for the sweep and their respective ranges or values to explore.
        sweep_config['parameters'] = {'learning_rate': {'values': [5e-3]}, 'reg': {'values': [1e-4]},
                                      'batch': {'values': [32]}}

        # Create a sweep project on wandb, which returns a unique identifier for the sweep.
        sweep_id = wandb.sweep(sweep_config, project='Purdue Face Recognition')
        # Start a wandb agent that will run the finetune function using configurations from the sweep.
        wandb.agent(sweep_id, finetune)
    else:
        # If wandb logging is not enabled, directly start the finetuning without hyperparameter tuning.
        finetune()

    # test('model_best_val_acc.pth')
