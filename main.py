
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.resnet import resnet34, resnet50
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from create_dataset import create_dataset
from utils.lr_schedule import inv_lr_scheduler
from utils.log_utils import ReDirectSTD

parser = argparse.ArgumentParser(description='SSDA Classification')

parser.add_argument('--steps', type=int, default=50000, metavar='N', help='maximum number of iterations to train (default: 50000)') # Step 수
parser.add_argument('--net', type=str, default='vgg', choices=['alexnet', 'vgg', 'resnet34', 'resnet50'], help='which network to use') # Feature extractor로 사용될 network 선택
parser.add_argument('--source', type=str, default='real', help='source domain')     # Source dataset 선택
parser.add_argument('--target', type=str, default='clipart', help='target domain')  # target dataset 선택

parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)') # learning rate
parser.add_argument('--bs', type=int, default=24, metavar='BS', help='Batch size (default | Alexnet:32, Others:24) ') # batch size ( S: x, T : x, uT : 2x)
parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.05)') # Temperature for sharpening softmax prediction
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM', help='value of lamda')         # parameter for minimax entropy loss
parser.add_argument('--save_check',type=bool, default=True, help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda', help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model')

parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office', 'office_home'], help='the name of dataset') # Select dataset
parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')                                  # The number of labeled target data
parser.add_argument('--patience', type=int, default=5, metavar='S', help='early stopping to wait for improvement before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True, help='early stopping on validation or not')
parser.add_argument('--log_file', type=str, default='./C2R_3shot.log', help='dir to save checkpoint')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    #log file 생성
    log_file_name = './logs/' + '/' + args.log_file
    ReDirectSTD(log_file_name, 'stdout', True)

    print('Dataset %s Source %s Target %s Labeled num per class %s Network %s'
          % (args.dataset, args.source, args.target, args.num, args.net))

    source_loader, target_loader, target_loader_unl,\
    target_loader_val, target_loader_test, class_list = create_dataset(args)
    record_dir = 'record/%s' % (args.dataset)

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    record_file = os.path.join(record_dir, 'net_{network}_{source}_to_{target}_num_{num_shot}'
                               .format(network=args.net, source=args.source, target=args.target, num_shot=args.num))

    # random seed 지정
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Feature extractor model 선택
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
        F1 = Predictor_deep(num_class=len(class_list), inc=inc, temp=args.T)
    else:
        F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

    G.to(device)
    F1.to(device)

    if os.path.exists(args.checkpath) == False:
        os.mkdir(args.checkpath)

    G.train()
    F1.train()

    optimizer_g = optim.SGD(G.parameters(), lr=1.0,
                            momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(F1.parameters(), lr=1.0,
                            momentum=0.9, weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(device)

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0
    counter = 0

    for step in range(args.steps):

        optimizer_g = inv_lr_scheduler(
            param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(
            param_lr_f, optimizer_f, step, init_lr=args.lr)

        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        im_data_s, gt_labels_s = data_s
        im_data_t, gt_labels_t = data_t
        im_data_tu = data_t_unl[0]

        gt_labels_s = gt_labels_s.long().to(device)
        gt_labels_t = gt_labels_t.long().to(device)
        im_data_s = im_data_s.to(device)
        im_data_t = im_data_t.to(device)
        im_data_tu = im_data_tu.to(device)

        zero_grad_all()

        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)

        output = G(data)
        out1 = F1(output)

        # cross entropy : labeled source, target data
        loss = criterion(out1, target)
        print(loss)

        loss.backward(retain_graph=True)

        optimizer_g.step()
        optimizer_f.step()

        zero_grad_all()

        # Minimax entropy : unlabeled target data
        output = G(im_data_tu)

        out_tu = F1(output, reverse=True, eta=1.0)
        out_tu = F.softmax(out_tu, dim=1)
        loss_tu = args.lamda * torch.mean(torch.sum(out_tu * (torch.log(out_tu + 1e-5)), 1))
        loss_tu.backward()

        optimizer_g.step()
        optimizer_f.step()

        log_train = 'S: {} → T: {} [Train] Ep: {} lr: {:.6f} ' \
                    '| Cross_entropy_loss: {:.6f}, Entropy_loss: {:.6f}'\
                    .format(args.source, args.target,
                            step, lr, loss.data, -loss_tu.data)

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0 and step > 0:

            loss_test, acc_test = test(target_loader_test, G, F1, class_list)
            loss_val, acc_val = test(target_loader_val, G, F1, class_list)

            G.train()
            F1.train()

            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break

            print('best acc test %f best acc val %f'
                  % (best_acc_test, acc_val))
            print('record %s' % record_file)

            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step, best_acc_test, acc_val))

            G.train()
            F1.train()

            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_to_{}_step_{}.pth.tar"
                                        .format(args.source, args.target, step)))
                torch.save(F1.state_dict(), os.path.join(args.checkpath, "F_iter_model_{}_to_{}_step_{}.pth.tar".format(args.source, args.target, step)))

def test(loader, G, F1, class_list):

    G.eval()
    F1.eval()

    test_loss = 0
    correct = 0
    size = 0
    
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t, gt_labels_t = data_t

            gt_labels_t = gt_labels_t.long().to(device)
            im_data_t = im_data_t.to(device)

            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred = output1.data.max(1)[1]

            for t, p in zip(gt_labels_t.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            correct += pred.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct, size, 100. * correct / size))

    return test_loss.data, 100. * float(correct) / size

if __name__ == '__main__':
    main()

