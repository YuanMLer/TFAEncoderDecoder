import torch
import numpy as np
from utils import AverageMeter

__all__ = ['train', 'test', 'superloss', 'test_return_data', 'test_return_attention_data']

def train(loader,model,optimizer,criterion,args):
    """
    train the model use the loader data
    :param loader: the dataloader
    :param model: the model which we use
    :param optimizer: the optimizer
    :param criterion:the loss function
    :param args: the parserArgs
    :return:
    """
    losses = AverageMeter()

    for idx,(history_msg,predict_M,predict_obs) in enumerate(loader):
        if torch.cuda.is_available():
            history_msg = history_msg.cuda()
            predict_M = predict_M.cuda()
            predict_obs = predict_obs.cuda()
        outs = model(history_msg, predict_M)
        # if idx % 100 == 0:
        #     print("history_msg = {}".format(history_msg))
        #     print("predict_M = {}".format(predict_M))
        #     print("predict_obs = {}".format(predict_obs))
        #     print("outs = {}".format(outs))
        # print("outs.size = {}".format(outs.size()))
        # print("predict_obs.size = {}".format(predict_obs.size()))
        loss = criterion(outs, predict_obs)
        n = predict_obs.size(0)*predict_obs.size(1)*predict_obs.size(2)
        losses.update(loss, n)
        optimizer.zero_grad()
        loss.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
    return losses

def test(loader,model,criterion,args):
    """
    test the model use the loader data
    :param loader: the dataloader
    :param model: the model which we use
    :param criterion:the loss function
    :param args: the parserArgs
    :return:
    """
    losses = AverageMeter()

    for idx,(history_msg,predict_M,predict_obs) in enumerate(loader):
        if torch.cuda.is_available():
            history_msg = history_msg.cuda()
            predict_M = predict_M.cuda()
            predict_obs = predict_obs.cuda()
        outs = model(history_msg,predict_M)
        loss = criterion(outs,predict_obs)
        n = predict_obs.size(0)*predict_obs.size(1)*predict_obs.size(2)
        losses.update(loss,n)
    return losses

def superloss(loader,criterion,args):
    """
    get the super computer loss
    :param loader:
    :param criterion:
    :param args: the parserArgs
    :return:
    """
    losses = AverageMeter()

    for idx, (history_msg, predict_M, predict_obs) in enumerate(loader):
        if torch.cuda.is_available():
            history_msg = history_msg.cuda()
            predict_M = predict_M.cuda()
            predict_obs = predict_obs.cuda()
        outs = predict_M[:,:,[1,4,3]]
        loss = criterion(outs, predict_obs)
        n = predict_obs.size(0) * predict_obs.size(1) * predict_obs.size(2)
        losses.update(loss, n)
    return losses


def test_return_data(loader, model, mse_criterion, _mse_criterion, args, getAttention=False):
    """
    test the model use the loader data,and return the .data include pred value,
    super calculate value and the observe value
    :param loader: the dataloader
    :param model: the model which we use
    :param mse_criterion:the loss function
    :param _mse_criterion:other loss function
    :param args: the parserArgs
    :return:
    """
    mse_losses = AverageMeter()
    num = 0
    _mse_losses = AverageMeter()
    predict_M_data = []
    obs_data = []
    pred_out_data = []
    attentions_data = []
    for idx, (history_msg, predict_M, predict_obs) in enumerate(loader):
        if idx % args.pred_len == 0:
            # print("idx = {}".format(idx))
            # print("args.pred_len = {}".format(args.pred_len))
            if torch.cuda.is_available():
                history_msg = history_msg.cuda()
                predict_M = predict_M.cuda()
                predict_obs = predict_obs.cuda()
            if getAttention:
                out, attentions = model(history_msg, predict_M, getAttention)
                for _id in range(len(attentions)):
                    attentions_data.append(attentions[_id])
            else:
                out = model(history_msg, predict_M, getAttention)

            # print("out.size = {}".format(out.size()))
            # print("predict_obs.size = {}".format(predict_obs.size()))
            # print("predict_M.size = {}".format(predict_M.size()))
            mse_losse = mse_criterion(out, predict_obs)
            _mse_losse = _mse_criterion(out, predict_obs)
            n = predict_obs.size(0) * predict_obs.size(1) * predict_obs.size(2)
            num += n
            mse_losses.update(mse_losse, n)
            _mse_losses.update(_mse_losse, n)

            if args.batch_first:
                out = out.squeeze(1)
                predict_M = predict_M.squeeze(1)
                predict_obs = predict_obs.squeeze(1)
            else:
                out = out.squeeze(0)
                predict_M = predict_M.squeeze(0)
                predict_obs = predict_obs.squeeze(0)
            # print("out.size = {}".format(out.size()))
            # print("predict_obs.size = {}".format(predict_obs.size()))
            # print("predict_M.size = {}".format(predict_M.size()))
            # print("predict_M.size = {}".format(predict_M.size()))
            predict_M = predict_M[:,(1, 4, 3)].cpu().detach().numpy()
            predict_obs = predict_obs.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            # print("predict_M = {}".format(predict_M))
            # print("predict_obs = {}".format(predict_obs))
            # print("out = {}".format(out))
            # print("len(predict_M) = {}".format(len(predict_M)))
            for id_ in range(len(predict_M)):
                predict_M_data.append(predict_M[id_].tolist())
                obs_data.append(predict_obs[id_].tolist())
                pred_out_data.append(out[id_].tolist())
            # print(predict_M_data)
        else:
            continue
    # print("obs_data = {}".format(obs_data))

    print("*" * 80)
    print("getAttention in test_return_data = {}".format(getAttention))

    if getAttention:
        return mse_losses.avg, _mse_losses.avg, num, predict_M_data, obs_data, pred_out_data, attentions_data
    else:
        return mse_losses.avg, _mse_losses.avg, num, predict_M_data, obs_data, pred_out_data

def test_return_dul_attention_data(loader, model, mse_criterion, _mse_criterion, args):
    """
    test the model use the loader data,and return the .data include pred value,
    super calculate value and the observe value
    :param loader: the dataloader
    :param model: the model which we use
    :param mse_criterion:the loss function
    :param _mse_criterion:other loss function
    :param args: the parserArgs
    :return:
    """
    mse_losses = AverageMeter()
    num = 0
    _mse_losses = AverageMeter()
    predict_M_data = []
    obs_data = []
    pred_out_data = []
    attention1_data = []
    attention2_data = []
    for idx, (history_msg, predict_M, predict_obs) in enumerate(loader):
        if idx % args.pred_len == 0:
            if torch.cuda.is_available():
                history_msg = history_msg.cuda()
                predict_M = predict_M.cuda()
                predict_obs = predict_obs.cuda()
            out, attention1_, attention2_ = model(history_msg, predict_M)

            for _id in range(len(attention1_)):
                attention1_data.append(attention1_[_id])
                attention2_data.append(attention2_[_id])

            mse_losse = mse_criterion(out, predict_obs)
            _mse_losse = _mse_criterion(out, predict_obs)
            n = predict_obs.size(0) * predict_obs.size(1) * predict_obs.size(2)
            num += n
            mse_losses.update(mse_losse, n)
            _mse_losses.update(_mse_losse, n)

            if args.batch_first:
                out = out.squeeze(1)
                predict_M = predict_M.squeeze(1)
                predict_obs = predict_obs.squeeze(1)
            else:
                out = out.squeeze(0)
                predict_M = predict_M.squeeze(0)
                predict_obs = predict_obs.squeeze(0)

            predict_M = predict_M[:, (1, 4, 3)].cpu().detach().numpy()
            predict_obs = predict_obs.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            for id_ in range(len(predict_M)):
                predict_M_data.append(predict_M[id_].tolist())
                obs_data.append(predict_obs[id_].tolist())
                pred_out_data.append(out[id_].tolist())
        else:
            continue
    return mse_losses.avg, _mse_losses.avg, num, predict_M_data, obs_data, pred_out_data,\
           attention1_data, attention2_data

def test_return_attention_data(loader, model, mse_criterion, _mse_criterion, args):
    """
    test the model use the loader data,and return the .data include pred value,
    super calculate value and the observe value
    :param loader: the dataloader
    :param model: the model which we use
    :param mse_criterion:the loss function
    :param _mse_criterion:other loss function
    :param args: the parserArgs
    :return:
    """
    mse_losses = AverageMeter()
    num = 0
    _mse_losses = AverageMeter()
    predict_M_data = []
    obs_data = []
    pred_out_data = []
    attention1_data = []
    attention2_data = []
    for idx, (history_msg, predict_M, predict_obs) in enumerate(loader):
        if idx % args.pred_len == 0:
            if torch.cuda.is_available():
                history_msg = history_msg.cuda()
                predict_M = predict_M.cuda()
                predict_obs = predict_obs.cuda()
            out, attention1_, attention2_ = model(history_msg, predict_M)

            for _id in range(len(attention1_)):
                attention1_data.append(attention1_[_id])
                attention2_data.append(attention2_[_id])

            mse_losse = mse_criterion(out, predict_obs)
            _mse_losse = _mse_criterion(out, predict_obs)
            n = predict_obs.size(0) * predict_obs.size(1) * predict_obs.size(2)
            num += n
            mse_losses.update(mse_losse, n)
            _mse_losses.update(_mse_losse, n)

            if args.batch_first:
                out = out.squeeze(1)
                predict_M = predict_M.squeeze(1)
                predict_obs = predict_obs.squeeze(1)
            else:
                out = out.squeeze(0)
                predict_M = predict_M.squeeze(0)
                predict_obs = predict_obs.squeeze(0)

            predict_M = predict_M[:, (1, 4, 3)].cpu().detach().numpy()
            predict_obs = predict_obs.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            for id_ in range(len(predict_M)):
                predict_M_data.append(predict_M[id_].tolist())
                obs_data.append(predict_obs[id_].tolist())
                pred_out_data.append(out[id_].tolist())
        else:
            continue
    return mse_losses.avg, _mse_losses.avg, num, predict_M_data, obs_data, pred_out_data,\
           attention1_data, attention2_data
