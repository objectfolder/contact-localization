import torch.optim as optim
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'OF_CNN_MFCC':
        from OF_CNN_MFCC import of_cnn_mfcc
        modality_list = args.modality_list
        model = of_cnn_mfcc.OF_CNN_MFCC(args,
                                          use_touch='touch' in modality_list,
                                          use_audio='audio' in modality_list).cuda()
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 'RANDOM':
        from RANDOM_MODEL import random_model
        model = random_model.RANDOM_MODEL()
        optimizer = None

    return model, optimizer
