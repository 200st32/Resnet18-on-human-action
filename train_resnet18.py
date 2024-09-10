import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from resnet18_torchvision import build_model
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib
import myutils


def main(learning_rate, num_epochs, batch_size, output_path, model_type, balance):

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print("cuda available:", use_cuda)
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Get data loader
    train_loader, val_loader, test_loader = myutils.getdata(batch_size=batch_size)

    # Create the model, optimizer, and loss function
    if model_type  == 'scratch':
        model = build_model(pretrained=False, fine_tune=True, num_classes=15).to(device) 

    elif model_type == 'torchvision':
        model = build_model(pretrained=True, fine_tune=False, num_classes=15).to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = nn.CrossEntropyLoss()

    # Train the model
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch_loss = myutils.train_model(model, train_loader, optimizer, loss_function, device)
        train_loss.append(train_epoch_loss)
        print(f"Train loss:{train_epoch_loss:.4f}")
        val_epoch_loss = myutils.val_model(model, val_loader, loss_function, device)
        val_loss.append(val_epoch_loss)
        print(f"Val loss:{val_epoch_loss:.4f}")
    try:
        # Plot the loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.num_epochs + 1), train_loss, marker='o', linestyle='-', color='b')
        plt.plot(range(1, args.num_epochs + 1), val_loss, marker='o', linestyle='-', color='g')
        plt.title(f'Training and validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)
        plt.savefig(f"{output_path}/{model_type}loss.png")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting acc and loss E: {e}")

    # run test data
    test_loss, test_acc = myutils.test_model(model, test_loader, loss_function, device)
    print(f"Accuracy on the test set: {test_acc:.3f}%")
    print(f"Loss on the test set: {test_loss:.2%}")

    #save model weight
    torch.save(model.state_dict(), f"{output_path}/{model_type}_weight.pt")

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Resnet18 Training Script')

    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--output_path',
                        type=str,
                        default="./myoutput",
                        help='Output path')
    parser.add_argument('--model_type',
                        type=str,
                        default="torchvision",
                        help='Model type')
    parser.add_argument('--balance',
                        action="store_true",
                        help='Use balance dataset')
    args = None
    args, unparsed = parser.parse_known_args()

    isExist = os.path.exists(args.output_path)
    if isExist == False:
        os.makedirs(args.output_path)
    main(args.learning_rate, args.num_epochs, args.batch_size, args.output_path, args.model_type, args.balance)






