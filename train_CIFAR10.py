import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vat import VATLoss
import data_utils
import utils

import torch
import torch.nn as nn
import numpy

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
# class Net(nn.Module):
#     def __init__(self, keep_prob_hidden=0.5, lrelu_a=0.1, top_bn=False):
#         super(Net, self).__init__()
#         self.keep_prob_hidden = keep_prob_hidden
#         self.lrelu_a = lrelu_a
#         self.top_bn = top_bn
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
#         self.bn8 = nn.BatchNorm2d(256)
#         self.conv9 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
#         self.bn9 = nn.BatchNorm2d(128)
#         self.fc = nn.Linear(128, 10)

#     def forward(self, x):
#         rng = np.random.RandomState(1234)

#         x = self.conv1(x)
#         x = nn.functional.leaky_relu(self.bn1(x), self.lrelu_a)
#         x = self.conv2(x)
#         x = nn.functional.leaky_relu(self.bn2(x), self.lrelu_a)
#         x = self.conv3(x)
#         x = nn.functional.leaky_relu(self.bn3(x), self.lrelu_a)
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = nn.functional.dropout(x, p=self.keep_prob_hidden, training=self.training) 

#         x = self.conv4(x)
#         x = nn.functional.leaky_relu(self.bn4(x), self.lrelu_a)
#         x = self.conv5(x)
#         x = nn.functional.leaky_relu(self.bn5(x), self.lrelu_a)
#         x = self.conv6(x)
#         x = nn.functional.leaky_relu(self.bn6(x), self.lrelu_a)
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = nn.functional.dropout(x, p=self.keep_prob_hidden, training=self.training)

#         x = self.conv7(x)
#         x = nn.functional.leaky_relu(self.bn7(x), self.lrelu_a)
#         x = self.conv8(x)
#         x = nn.functional.leaky_relu(self.bn8(x), self.lrelu_a)
#         x = self.conv9(x)
#         x = nn.functional.leaky_relu(self.bn9(x), self.lrelu_a)
#         x = tf.reduce_mean(x, reduction_indices=[1, 2])  # Global average pooling
#         x = self.fc(h, 128, 10, seed=rng.randint(123456), name='fc')

#         if FLAGS.top_bn:
#             x = self.bn(x, 10, is_training=is_training,
#                     update_batch_stats=update_batch_stats, name='bfc')

#         return h


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x


def train(args, model, device, data_iterators, optimizer):
    model.train()
     # Initialize lists for storing training statistics
    train_ce_losses = []
    train_vat_losses = []
    train_prec1 = []   
    valid_prec1 = []
    best_valid_acc = 0.0
    patience = 5  # maximum number of epochs without improvement
    epochs_since_last_improvement = 0
    

    for i in tqdm(range(args.iters)):
        model.train()
        # reset
        if i % args.log_interval == 0:
            ce_losses = utils.AverageMeter()
            vat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
            
            
        x_l, y_l = next(data_iterators['labeled'])
        x_ul, _ = next(data_iterators['unlabeled'])

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()

        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        cross_entropy = nn.CrossEntropyLoss()

        lds = vat_loss(model, x_ul)
        output = model(x_l)
        # # Compute the softmax probabilities
        # probs = F.softmax(output, dim=1)

        # # Compute the log-softmax probabilities
        # log_probs = F.log_softmax(output, dim=1)

        # # Compute the entropy
        # entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))

        classification_loss = cross_entropy(output, y_l)
        loss = classification_loss + args.alpha * lds #+ 0.01 * entropy
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        vat_losses.update(lds.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])

        if i % args.log_interval == 0:
            print(f'\nIteration: {i}\t'
                f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
                f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')
                # Validation
            val_dataloader = DataLoader(data_iterators['val'].dataset, shuffle=False)
            valid_acc = test(model, device, val_dataloader)

            #valid_losses.append(valid_loss)
            valid_prec1.append(valid_acc)
            print(f'\nEpoch: {i}\tValidation Precision {valid_acc:.3f}')
            
            # check for improvement and update best weights
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                epochs_since_last_improvement = 0
                torch.save(model.state_dict(), 'best_weights.pth')
            else:
                epochs_since_last_improvement += 1

            # check if we should stop early
            if epochs_since_last_improvement > patience:
                print(f'Early stopping at epoch {i}, using best weights from epoch {i-patience}')
                break
        # Append training statistics to lists
        train_ce_losses.append(ce_losses.avg)
        train_vat_losses.append(vat_losses.avg)
        train_prec1.append(prec1.avg)
 
        
    
        
    # Plot training and validation statistics
    plt.plot(train_prec1)
    plt.plot(valid_prec1)
    plt.title('Precision')
    plt.xlabel('Iteration')
    plt.ylabel('Precision')
    plt.legend(['Train', 'Validation'])
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(train_ce_losses, label='Train Cross Entropy Loss')
    ax.plot(train_vat_losses, label='Train VAT Loss')
    #ax.plot(valid_losses, label='Validation Loss')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Value')
    ax.set_title('Training and Validation Metrics')
    ax.legend()

    plt.show()
   

def test(model, device, data_iterators):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(data_iterators): 

            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            correct += torch.eq(outputs.max(dim=1)[1], y).detach().cpu().float().sum()
            

        test_acc = correct / len(data_iterators.dataset) * 100.

    return test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=10000, metavar='N',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_iterators = data_utils.get_iters(
        dataset='MNIST',
        root_path='.',
        l_batch_size=args.batch_size,
        ul_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, model, device, data_iterators, optimizer)
    # Create a new instance of your model
    model = Net().to(device)

    # Load the state dictionary of the best model
    best_model_state_dict = torch.load("/home/nassimiheb/DL/VAT-pytorch/best_weights.pth")
    model.load_state_dict(best_model_state_dict)
    test_accuracy = test(model, device, data_iterators['test'])
    print(f'\nTest Accuracy: {test_accuracy :.4f}%\n')

if __name__ == '__main__':
    main()
