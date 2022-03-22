from __future__ import print_function


import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from create_dataset import create_dataset_test
from plot_confusion_matrix import plot_confusion_matrix

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')

parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=12000, metavar='step', help='loading step')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda', help='dir to save checkpoint')
parser.add_argument('--output', type=str, default='./output.txt', help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B', help='which network ')
parser.add_argument('--source', type=str, default='real', metavar='B', help='board dir')
parser.add_argument('--target', type=str, default='clipart', metavar='B', help='board dir')
parser.add_argument('--dataset', type=str, default='multi', choices=['multi'], help='the name of dataset, multi is large scale dataset')

parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')
parser.add_argument('--topk', type=int, default=5, help='top-k')
parser.add_argument('--bs', type=int, default=24, metavar='BS', help='Batch size')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    print('dataset %s source %s target %s network %s' % (args.dataset, args.source, args.target, args.net))
    target_loader_unl, class_list = create_dataset_test(args)

    if args.net == 'resnet34':
        G = resnet34()
        inc = 512
    elif args.net == 'resnet50':
        G = resnet50()
        inc = 2048
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    if "resnet" in args.net:
        F1 = Predictor_deep(num_class=len(class_list), inc=inc)
    else:
        F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

    G.to(device)
    F1.to(device)

    G.load_state_dict(torch.load(os.path.join(args.checkpath, "G_iter_model_{}_to_{}_step_{}.pth.tar" .format(args.source, args.target, args.step))))
    F1.load_state_dict(torch.load(os.path.join(args.checkpath, "F_iter_model_{}_to_{}_step_{}.pth.tar" .format(args.source, args.target, args.step))))


    if os.path.exists(args.checkpath) == False:
        os.mkdir(args.checkpath)

    evaluate(target_loader_unl, class_list, G, F1, args.step, args.topk, output_file="%s_%s.txt" % (args.net, args.step))


def evaluate(loader, class_list, G, F1, step, topk, output_file="output.txt"):

    G.eval()
    F1.eval()
    size = 0
    correct = 0
    correct_topk = 0

    output_all = np.zeros((0, len(class_list)))
    confusion_matrix = torch.zeros(len(class_list), len(class_list))

    with open(output_file, "w") as f:
        with torch.no_grad():
            for _, (im_data_t, gt_labels_t, paths) in tqdm(enumerate(loader)):

                im_data_t, gt_labels_t = im_data_t.to(device), gt_labels_t.long().to(device)
                feat = G(im_data_t)
                output = F1(feat)
                output_all = np.r_[output_all, output.data.cpu().numpy()]

                size += im_data_t.size(0)
                pred = output.data.max(1)[1]
                _, pred_topk = torch.topk(output.data, topk, dim=-1)

                for t, p in zip(gt_labels_t.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                correct += pred.eq(gt_labels_t.data).cpu().sum()
                correct_topk += pred_topk.eq(gt_labels_t.view(-1, 1).expand_as(pred_topk)).cpu().sum()

                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred[i]))

            plot_confusion_matrix(confusion_matrix, class_list, step, class_name=False)

            print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, size, 100. * correct / size))
            print('Top {} Accuracy: {}/{} ({:.0f}%)\n'.format(topk, correct_topk, size, 100. * correct_topk / size))


if __name__ == '__main__':
    main()

