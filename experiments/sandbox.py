from glob import glob
import wandb
import hashlib
import os, json, sys
import argparse
from random import shuffle
from sklearn.metrics import accuracy_score, f1_score

import torch, random, torch.nn as nn, numpy as np
from torch import optim
from torch.cuda.amp import GradScaler
import lightning.pytorch as pl

from game_parser import GameParser
from model import Model

from mineclip import MineCLIP
from omegaconf import OmegaConf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Where the data and model weight folders are stored
scratch_dir = '/scratch/eecs692w23_class_root/eecs692w23_class/anrao/'

# Given label number x and total labels n, convert into a one hot vector of shape (1,n)
def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

# Statistics
def print_epoch(data,acc_loss,lst, val=False):
    print(f'{acc_loss/len(lst):9.4f}',end='; ',flush=True)
    data = list(zip(*data))
    val_str = "val_" if val else ""
    print("Predictions and GT: ", data)
    for x in data:
        a, b = list(zip(*x))
        if max(a) <= 1:
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{sum(a)/len(a):5.3f},{sum(b)/len(b):5.3f},{len(b)})', end=' ',flush=True)
        else:
            print(f'({accuracy_score(a,b):5.3f},{f1_score(a,b,average="weighted"):5.3f},{len(b)})', end=' ',flush=True)
            wandb.log({val_str+"acc": accuracy_score(a,b), val_str+"acc_loss": acc_loss/len(lst), val_str+"f1":f1_score(a,b,average="weighted")})
    print('', end='; ',flush=True)

# Make splits for training and evaluation procedures. Shouldn't care much here.
def make_splits():
    if not os.path.isfile('dataset_splits.json'):
        dirs = sorted(glob(scratch_dir+'*_logs/*'))
        games = sorted(list(map(GameParser, dirs)), key=lambda x: len(x.question_pairs), reverse=True)

        # Using python stepping to split data in 20-20-60 form
        test = games[0::5]
        val = games[1::5]
        train = games[2::5]+games[3::5]+games[4::5]

        dataset_splits = {'test' : [g.game_path for g in test], 'validation' : [g.game_path for g in val], 'training' : [g.game_path for g in train]}
        json.dump(dataset_splits, open('dataset_splits.json','w'))
    else:
        dataset_splits = json.load(open('dataset_splits.json'))

    return dataset_splits

def construct_gt_pred(l, game, exp):
    prediction = []
    ground_truth = []
    data = []
    for gt, prd in l:
        # lbls = [int(a==b) for a,b in zip(gt[0],gt[1])]
        lbls = np.equal(gt[0],gt[1]).astype(int).tolist()
        lbls += [['NO', 'MAYBE', 'YES'].index(gt[0][0]),['NO', 'MAYBE', 'YES'].index(gt[0][1])]
        if gt[0][2] in game.materials_dict:
            lbls.append(game.materials_dict[gt[0][2]])
        else:
            lbls.append(0)
        lbls += [['NO', 'MAYBE', 'YES'].index(gt[1][0]),['NO', 'MAYBE', 'YES'].index(gt[1][1])]
        if gt[1][2] in game.materials_dict:
            lbls.append(game.materials_dict[gt[1][2]])
        else:
            lbls.append(0)
        prd = prd[exp:exp+1]
        lbls = lbls[exp:exp+1]
        data.append([(g,torch.argmax(p).item()) for p,g in zip(prd,lbls)])

        # Mainly for experiments 0,1 and 2; Not relevant here.
        # p, g = zip(*[(p,torch.eye(p.shape[0]).float()[g]) for p,g in zip(prd,lbls)])
        if exp == 0:
            pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==0 or (random.random() < 2/3)]))
        elif exp == 1:
            pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==0 or (random.random() < 5/6)]))
        elif exp == 2:
            pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls) if gt==1 or (random.random() < 5/6)]))
        else:
            pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls)]))
            #pairs = torch.cat((prd, lbls), dim=1).tolist()
        if pairs:
            p,g = pairs
        else:
            continue
        prediction.append(torch.cat(p))
        ground_truth += g

    if prediction:
        prediction = torch.stack(prediction)
    if ground_truth:
        ground_truth = torch.tensor(ground_truth).long().to(DEVICE)
    return prediction, ground_truth, data

# Model training and evaluation procedures
def do_split(model,lst,exp,criterion,optimizer=None,global_plan=False, player_plan=False, clip_flag=False, scheduler = None):
    skipped = 0
    data = []
    acc_loss = 0
    loss = 0
    l = None
    scaler = GradScaler()

    accum_steps = 8

    for i_game, game in enumerate(lst):
        # Model file's function returns a list of ground truths and their associated predictions
        l = model(game, global_plan=global_plan, player_plan=player_plan,clip=clip_flag)
        prediction, ground_truth, data_temp = construct_gt_pred(l, game, exp)
        data += data_temp

        if prediction == None or ground_truth == None or len(prediction) == 0 or len(ground_truth) == 0:
            print("No predictions or truth found")
            skipped += 1
            continue
        loss = criterion(prediction,ground_truth)

        if model.training and (not optimizer is None):
            loss.backward()

            if (i_game+1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()

            if i_game==len(lst)-1 and len(lst) % accum_steps != 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
        acc_loss += loss.item()
    print_epoch(data,acc_loss,lst, val=False if model.training else True)
    print(f"Skipped {skipped} games of {len(lst)} this epoch.")
    return acc_loss, data

def main(args):
    print(args)
    print(f'PID: {os.getpid():6d}')

    if args.seed=='Random':
        pass
    elif args.seed=='Fixed':
        random.seed(0)
        torch.manual_seed(1)
    else:
        print('Seed must be in [Random, Fixed], but got',args.seed)
        exit()

    dataset_splits = make_splits()

    if args.use_dialogue=='Yes':
        d_flag = True
    elif args.use_dialogue=='No':
        d_flag = False
    else:
        print('Use dialogue must be in [Yes, No], but got',args.use_dialogue)
        exit()

    # Flag to know when we want to train the model
    train_flag = args.train

    if not args.experiment in list(range(9)):
        print('Experiment must be in',list(range(9)),', but got',args.experiment)
        exit()

    if args.seq_model=='GRU':
        seq_model = 0
    elif args.seq_model=='LSTM':
        seq_model = 1
    elif args.seq_model=='Transformer':
        seq_model = 2
    elif args.seq_model=='X-Transformer':
        seq_model = 3
        print("Loaded X Transformer", flush=True)
    else:
        print('The sequence model must be in [GRU, LSTM, Transformer], but got', args.seq_model)
        exit()

    if args.plans=='Yes':
        global_plan = (args.pov=='Third') or ((args.pov=='None') and (args.experiment in list(range(3))))
        player_plan = (args.pov=='First') or ((args.pov=='None') and (args.experiment in list(range(3,9))))
    elif args.plans=='No' or args.plans is None:
        global_plan = False
        player_plan = False
    else:
        print('Use Plan must be in [Yes, No], but got',args.plan)
        exit()
    print('global_plan', global_plan, 'player_plan', player_plan)

    clip = None
    if args.clip:
        cfg = OmegaConf.load("conf.yaml")
        OmegaConf.set_struct(cfg, False)
        ckpt = cfg.pop("ckpt")
        OmegaConf.set_struct(cfg, True)
        clip = MineCLIP(**cfg).to(DEVICE)
        clip.load_ckpt(ckpt.path, strict=True)
        clip.eval()
        assert (
        hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
        ), "broken ckpt"
        print("Loaded MineCLIP")

    if train_flag:
        if args.pov=='None':
            val    = [GameParser(f,d_flag,0,clip) for f in dataset_splits['validation']]
            train  = [GameParser(f,d_flag,0,clip) for f in dataset_splits['training']]
            if args.experiment > 2:
                val   += [GameParser(f,d_flag,4,clip) for f in dataset_splits['validation']]
                train += [GameParser(f,d_flag,4,clip) for f in dataset_splits['training']]
        elif args.pov=='Third':
            val    = [GameParser(f,d_flag,3,clip) for f in dataset_splits['validation']]
            train  = [GameParser(f,d_flag,3,clip) for f in dataset_splits['training']]
        elif args.pov=='First':
            val    = [GameParser(f,d_flag,1,clip) for f in dataset_splits['validation']]
            train  = [GameParser(f,d_flag,1,clip) for f in dataset_splits['training']]
            val   += [GameParser(f,d_flag,2,clip) for f in dataset_splits['validation']]
            train += [GameParser(f,d_flag,2,clip) for f in dataset_splits['training']]
        else:
            print('POV must be in [None, First, Third], but got', args.pov)
            exit()

    model = Model(seq_model).to(DEVICE)

    print(model)
    if train_flag:
        model.train()

    learning_rate = 1e-5
    weight_decay = 1e-4
    num_epochs = 1000#1#

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    wandb.init(
        # set the wandb project where this run will be logged
        project="MindCraft",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": args.seq_model,
            "Experiment Number": args.experiment,
            "weight_decay": weight_decay,
            "scheduler": "None",
            "CLIP": "Yes",
            "Video": "Yes",
            "CLIP-Agg": "Attn",
        })
    print(str(criterion), str(optimizer))

    min_acc_loss = 100
    max_f1 = 0
    epochs_since_improvement = 0
    wait_epoch = 40
    scheduler = None

    if train_flag:
        for epoch in range(num_epochs):
            print(f'{os.getpid():6d} {epoch+1:4d},',end=' ', flush=True)
            print(f"Epoch {epoch} Training:")
            shuffle(train)
            model.train()
            do_split(model,train,args.experiment,criterion,optimizer=optimizer,global_plan=global_plan, player_plan=player_plan, clip_flag=args.clip, scheduler=scheduler)
            print(f"Epoch {epoch} Validation:")
            model.eval()
            acc_loss, data = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, clip_flag=args.clip)

            data = list(zip(*data))
            for x in data:
                a, b = list(zip(*x))
            f1 = f1_score(a,b,average='weighted')
            if (max_f1 < f1):
                max_f1 = f1
                epochs_since_improvement = 0
                print('^')
                torch.save(model.cpu().state_dict(), args.save_path)
                model = model.to(DEVICE)
            else:
                epochs_since_improvement += 1
                print()
            # if (min_acc_loss > acc_loss):
            #     min_acc_loss = acc_loss
            #     epochs_since_improvement = 0
            #     print('^')
            # else:
            #     epochs_since_improvement += 1
            #     print()

            #if epochs_since_improvement > 10:
            if epoch > wait_epoch and epochs_since_improvement > 20:
                break
    print()
    print('Test')
    model.load_state_dict(torch.load(args.save_path))

    val = None
    train = None

    if args.pov=='None':
        test = [GameParser(f,d_flag,0,clip) for f in dataset_splits['test']]
        if args.experiment > 2:
            test += [GameParser(f,d_flag,4,clip) for f in dataset_splits['test']]
    elif args.pov=='Third':
        test = [GameParser(f,d_flag,3,clip) for f in dataset_splits['test']]
    elif args.pov=='First':
        test  = [GameParser(f,d_flag,1,clip) for f in dataset_splits['test']]
        test += [GameParser(f,d_flag,2,clip) for f in dataset_splits['test']]
    else:
        print('POV must be in [None, First, Third], but got', args.pov)

    clip_split(clip, test, criterion)
    model.eval()
    _, data = do_split(model,test,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, clip_flag=True)

    print()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pov', type=str,
                    help='point of view [None, First, Third]')
    parser.add_argument('--use_dialogue', type=str,
                    help='Use dialogue [Yes, No]')
    parser.add_argument('--plans', type=str,
                    help='Use plans [Yes, No]')
    parser.add_argument('--seq_model', type=str,
                    help='point of view [GRU, LSTM, Transformer]')
    parser.add_argument('--experiment', type=int,
                    help='point of view [0:AggQ1, 1:AggQ2, 2:AggQ3, 3:P0Q1, 4:P0Q2, 5:P0Q3, 6:P1Q1, 7:P1Q2, 8:P1Q3]')
    parser.add_argument('--save_path', type=str,
                    help='path where to save model')
    parser.add_argument('--seed', type=str,
                    help='Use random or fixed seed [Random, Fixed]')
    parser.add_argument('--train',  action='store_true',
                    help='Training')
    parser.add_argument('--clip',  action='store_true',
                    help='Flag to enable clip embeddings to be used')

    main(parser.parse_args())
