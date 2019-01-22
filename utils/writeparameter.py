import os
from .parserlist import ParserList
args,state = ParserList()

__all__ = ['writeparameter']
def writeparameter(checkpoint_path,model,args=args,state=state):
    """
    write the paremeter to the checkpoint path
    :param checkpoint_path: where to write
    :param model: model
    :param args:
    :param state:
    :return: None
    """
    with open(os.path.join(checkpoint_path, 'parameters.txt'), 'w') as f:
        f.write(' Total params:%.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        f.write("epochs = {}\n".format(args.epoches))
        f.write("batch-size = {}\n".format(args.batch_size))
        f.write("learning-rate = {}\n".format(args.lr))
        f.write("schedule = {}\n".format(args.schedule))
        f.write("ARCH = {}\n".format(args.arch))
        f.write("LOSS = {}\n".format(args.criterion))
        f.write("Optimizer = {}\n".format(args.optimizer))
        f.write("-" * 70)
        f.write("\nbidirectional = {}\n".format(args.bidirectional))
        f.write("natural-encoder-in-dim = {}\n".format(args.natural_encoder_in_dim))
        f.write("encoder-in-dim = {}\n".format(args.encoder_in_dim))
        f.write("encoder-hidden-dim = {}\n".format(args.encoder_hidden_dim))
        f.write("encoder-num-layer = {}\n".format(args.encoder_num_layer))
        f.write("natural-decoder-in-dim = {}\n".format(args.natural_decoder_in_dim))
        # f.write("decoder-in-dim = {}\n".format(args.decoder_in_dim))
        f.write("decoder-hidden-dim = {}\n".format(args.decoder_hidden_dim))
        f.write("decoder-num-layer = {}\n".format(args.decoder_num_layer))
        f.write("decoder-out-dim = {}\n".format(args.decoder_out_dim))
        f.write("seq-len = {}\n".format(args.seq_len))
        f.write("pred-len = {}\n".format(args.pred_len))
        f.write(str(state))