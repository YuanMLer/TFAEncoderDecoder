# -*- encoding = utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import models as models
from utils import ParserList,AverageMeter,Logger,mkdir_p,AverageMeter,\
    init_params,adjust_learning_rate,save_checkpoint,writeparameter
from dataset import WeatherDataset,WeatherDatasetMoreData
from losses import TotalVarMSEloss
from trainandtest import *
from test import *


# Get Parameters;args: a constant;state: a variant we can change the value,it is a map variant
args,state = ParserList()
state = sorted(state)
print(state)

# Use Cuda
use_cuda = torch.cuda.is_available()

# set the default data type
torch.set_default_tensor_type('torch.FloatTensor')

# Random seed
if args.manual_seed is None:
    args.manual_seed = 100
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)
if use_cuda:
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)

# reset dataset csv files
if args.rewrite_dataset_files:
    load_dataset(args)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_list = [ 'purelstm',  'taendecoder',
              'saendecoder', 'tsaendecoder']
bidirectional = 2 if args.bidirectional else 1
model_path = args.arch+"_EL_"+str(args.encoder_num_layer)+"_DL_"+str(args.decoder_num_layer)+\
             "_"+args.criterion+"_" +args.optimizer+"_SEQ_"+str(args.seq_len)+"_PRED_"+str(args.pred_len)+\
             "_HD_"+str(args.encoder_hidden_dim)+"_BD_"+str(bidirectional)
station = range(0, 5)

def main():
    # refine  the single station
    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'TotalVarMSEloss':
        criterion = TotalVarMSEloss()
    print("we are going to use the modle:[{}]; Train single ?:[{}]".format(args.arch,args.train_single))
    print("id = {}".format(os.getpid()))
    if args.train_single:
        for i in station:
            print("*"*20+"station:[{}]".format(i)+"*"*20)
            train_root_dir = os.path.join(args.basedir, args.m_traindir_base, str(i))
            validation_root_dir = os.path.join(args.basedir, args.m_validationdir_base, str(i))
            test_root_dir = os.path.join(args.basedir, args.m_testdir_base, str(i))
            checkpoint_path = os.path.join(args.basedir, args.checkpoint,model_path,str(i))
            framework(train_root_dir, validation_root_dir, test_root_dir, checkpoint_path, criterion, args,sta=i)
    else:
        # train the main model use all the stations' data
        print("*" * 20 + "train main model" + "*" * 20)
        train_root_dir = os.path.join(args.basedir, args.traindir)
        validation_root_dir = os.path.join(args.basedir, args.validationdir)
        test_root_dir = os.path.join(args.basedir, args.testdir)
        checkpoint_path = os.path.join(args.basedir, args.checkpoint,model_path,'main')
        framework(train_root_dir, validation_root_dir, test_root_dir, checkpoint_path, criterion, args,sta=-1 )

# do the train eval and test process
def train_eval_test(model,lr,optimizer,criterion,train_loader,validation_loader,test_loader,sta,checkpoint_path,
                    start_epoch,args,super_train_loss,super_validation_loss,super_test_loss,best_loss):
    # todo add the eval process and add the test process
    if args.resume:
        epoches = args.epoches
    else:
        epoches = args.refine_epoches
    for epoch in range(start_epoch, epoches):
        lr = adjust_learning_rate(optimizer, epoch, args.adjust_per_epochs, lr, args.schedule,
                                  gamma=args.gamma)
        print("*" * 58)
        print("PID:[{}] | Sta:[{}] | model:[{}] | seq_len:[{}] | pred_len:[{}]".
              format(os.getpid(), sta, args.arch, args.seq_len, args.pred_len))
        print("Epoch:[%d | %d]; LR: [%f]" % (epoch, args.refine_epoches, lr))
        train_loss = train(train_loader, model, optimizer, criterion, args)
        print("train loss=[%.2f];\tsuper_train_loss=[%.2f]"%(train_loss.avg, super_train_loss.avg))
        validation_loss = test(validation_loader, model, criterion, args)
        print("validation loss=[%.2f];\tsuper_validation_loss=[%.2f]"%(validation_loss.avg, super_validation_loss.avg))
        test_loss = test(test_loader, model, criterion, args)
        print("test loss=[%.2f];\tsuper_test_loss=[%.2f]"%(test_loss.avg, super_test_loss.avg))
        # save checkpoints
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title="{} Log".format(args.arch), resume=True)
        logger.append([epoch, lr, train_loss.avg, validation_loss.avg, test_loss.avg,super_train_loss.avg,
                       super_validation_loss.avg,super_test_loss.avg])
        is_best = test_loss.avg < best_loss
        best_loss = min(test_loss.avg, best_loss)
        save_checkpoint({
            'lr': lr,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_static_dict': optimizer.state_dict(),
            'train_loss': train_loss.avg,
            'test_loss': test_loss.avg,
            'best_loss': best_loss
        },
            is_best=is_best,
            checkpoint_path=checkpoint_path,
            filename=args.resume_file)

def framework(train_root_dir, validation_root_dir,test_root_dir, checkpoint_path, criterion, args, sta):
    start_epoch = args.start_epoch  # start from epoch 0 or the last checkpoint epoch
    best_loss = np.inf  # best test accuracy
    lr = args.lr
    # Data loader
    train_set = WeatherDatasetMoreData(train_root_dir, args)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers)
    validation_set = WeatherDatasetMoreData(validation_root_dir, args)
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.workers)
    test_set = WeatherDatasetMoreData(test_root_dir, args)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)

    if not os.path.exists(checkpoint_path):
        mkdir_p(checkpoint_path)
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch in model_list:
        model = models.__dict__[args.arch](args=args)
    else:
        print("*"*20+"please add the model into the model list"+"*"*20)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        model = model.cuda()

    print(' Total params:%.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    title = 'Weather precdict dataset' + args.arch
    super_train_loss = superloss(train_loader, criterion, args)
    super_validation_loss = superloss(validation_loader, criterion, args)
    super_test_loss = superloss(test_loader, criterion, args)
    # resume
    resume_file = os.path.join(checkpoint_path, args.resume_file)
    if args.resume and os.path.isfile(resume_file):
        # Load checkpoint
        print("resume from checkpoint...")
        checkpoint = torch.load(resume_file)
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_static_dict'])
        # logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=False)
        logger.set_names(['Epoch', 'Learning Rate', 'Train losses', 'Valid losses', 'Test losses',
                          'Supper Train losses', 'Supper Validation losses', 'Supper Test losses'])
        writeparameter(checkpoint_path, model, args, state)
    # train eval and test
    train_eval_test(model, lr, optimizer, criterion, train_loader, validation_loader, test_loader,
                    sta, checkpoint_path, start_epoch, args, super_train_loss, super_validation_loss,
                    super_test_loss, best_loss)

    # test only
    if args.test:
        print("\nEvaluation only")
        test_loss = test(test_loader, model, criterion, args)
        print(' Test Loss:  %.8f' % (test_loss))
        # todo test, predict result and give the metric value
        return


if __name__ == '__main__':
    main()