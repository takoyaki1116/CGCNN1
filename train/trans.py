import os
import sys
import warnings
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.pardir)
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

from train import train
from validate import validate

from module.arguments import arguments
from module.function import *
from module.normalizer import Normalizer

args = arguments(trans=True)

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def trans():
    global args, best_mae_error

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task ==
                                                       'classification' else False)

    # pretrained model path
    model_a_path='../pre-trained/research-model/bulk_moduli-model_best.pth.tar'
    model_b_path='../pre-trained/research-model/sps-model_best.pth.tar'

    # load latest model state
    ckpt_a = torch.load(model_a_path)
    ckpt_b = torch.load(model_b_path)

    # load model
    if args.modelpath:
        if os.path.isfile(args.modelpath):
            print("=> loading model params '{}'".format(args.modelpath))
            model_checkpoint = torch.load(args.modelpath,
                                        map_location=lambda storage, loc: storage)
            print("=> loaded model params '{}'".format(args.modelpath))
        else:
            print("=> no model params found at '{}'".format(args.modelpath))
    else:
        print("=> modelpath wasn't set up")
        print("=> pretrained model will be loaded, instead")
        if args.use == 'bulk_moduli':
            model.load_state_dict(ckpt_a['state_dict'])
            print("=> pretrained bulk_moduli model was loaded")
        else:
            model.load_state_dict(ckpt_b['state_dict'])
            print("=> pretrained sps moduli prediction model was loaded")

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # fix the parameters
    params_to_update = []
    update_param_names = ['fc_out.weight', 'fc_out.bias', 'conv_to_fc.bias', 'conv_to_fc.weight']
    for name, param in model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(name)
            else:
                param.requires_grad= False

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)
                            
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(args, val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()
    
        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('../result/model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(args, test_loader, model, criterion, normalizer, test=True, trans=True)

if __name__ == '__main__':
    trans()
