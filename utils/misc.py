'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import shutil


__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'adjust_learning_rate','save_checkpoint','save_modles','AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def adjust_learning_rate(optimizer, epoch, adjust_per_epochs, lr, schedule, gamma=0.1):
    if schedule is None:
        if (epoch + 1) % adjust_per_epochs == 0:
            lr = lr * gamma
    else:
        if (epoch + 1) in schedule:
            lr = lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# save parameter
def save_checkpoint(state,is_best,checkpoint_path='checkpoint',filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_path):
        mkdir_p(checkpoint_path)
    filepath = os.path.join(checkpoint_path, filename)
    torch.save(state,filepath)
    if is_best:
        print("save the best modle")
        shutil.copy(filepath, os.path.join(checkpoint_path, 'model_best.tar'))


#  save all model datas
def save_modles(encoder, decoder, encoder_optimizer, decoder_optimizer, is_best, checkpoint="checkpoint",
                    encoderfilename="encoder.pkl", decoderfilename="decoder.pkl", encoderoptfile="encoderopt.pkl",
                    decoderoptfile="decoderopt.pkl"):
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    encoderfilepath = os.path.join(checkpoint, encoderfilename)
    decoderfilepath = os.path.join(checkpoint,decoderfilename)
    encoderoptfilepath = os.path.join(checkpoint,encoderoptfile)
    decoderoptfilepath = os.path.join(checkpoint,decoderoptfile)


    torch.save(encoder,encoderfilepath)
    torch.save(decoder,decoderfilepath)
    torch.save(encoder_optimizer,encoderoptfilepath)
    torch.save(decoder_optimizer,decoderoptfilepath)
    if is_best:
        print("save the best modle")
        shutil.copy(encoderfilepath, os.path.join(checkpoint, 'encoder_best.pkl'))
        shutil.copy(decoderfilepath, os.path.join(checkpoint, 'decoder_best.pkl'))
        shutil.copy(encoderoptfilepath, os.path.join(checkpoint, 'encoder_opt_best.pkl'))
        shutil.copy(decoderoptfilepath, os.path.join(checkpoint, 'decoder_opt_best.pkl'))




class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


