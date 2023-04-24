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
from clipstylemodel import Model

from mineclip import MineCLIP
from omegaconf import OmegaConf
#MODE = 'finetune_embed'
MODE = 'classic'
#MODE = 'zero_shot'
print(f"Performing {MODE} contrastive training")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Where the data and model weight folders are stored
scratch_dir = '/scratch/eecs692w23_class_root/eecs692w23_class/anrao/'

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
            #wandb.log({val_str+"acc": accuracy_score(a,b), val_str+"acc_loss": acc_loss/len(lst), val_str+"f1":f1_score(a,b,average="weighted")})
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
        lbls += [['YES', 'MAYBE', 'NO'].index(gt[0][0]),['YES', 'MAYBE', 'NO'].index(gt[0][1])]
        if gt[0][2] in game.materials_dict:
            lbls.append(game.materials_dict[gt[0][2]])
        else:
            lbls.append(0)
        lbls += [['YES', 'MAYBE', 'NO'].index(gt[1][0]),['YES', 'MAYBE', 'NO'].index(gt[1][1])]
        if gt[1][2] in game.materials_dict:
            lbls.append(game.materials_dict[gt[1][2]])
        else:
            lbls.append(0)

        prd = prd[exp:exp+1]
        lbls = lbls[exp:exp+1]
        pairs = list(zip(*[(pr,gt) for pr,gt in zip(prd,lbls)]))
        data.append([(g,torch.argmax(p.softmax(dim=-1).detach().cpu())) for p,g in zip(prd,lbls)])

        if pairs:
            p,g = pairs
        else:
            continue
        prediction.append(p)
        ground_truth += g

    if ground_truth:
        ground_truth = torch.tensor(ground_truth).long().to(DEVICE)
    return prediction, ground_truth, data

def clip_split(model, clip_model, lst, exp, criterion, optimizer):
    data = []
    for i_game, game in enumerate(lst):
        _,d,q,f,c_f = zip(*list(game))
        l, questions, video_feats = model(game, global_plan=True, player_plan=True,clip=True)
        print(video_feats)
        acc_loss = 0
        embedding, ground_truth, data_temp = construct_gt_pred(l, game, exp)

        exp_indexer = {3:0,4:1,5:2,6:0,7:1,8:2}
        loss = 0
        pred = []
        for q_i, question in enumerate(questions):
            if MODE == "zero_shot":
                temp_embed = torch.stack([torch.Tensor(video_feats[q_i])], 0)
                print(clip_model.forward_reward_head(temp_embed.to(DEVICE), text_tokens=question[exp_indexer[exp]]))
                pred.append(clip_model.forward_reward_head(temp_embed.to(DEVICE), text_tokens=question[exp_indexer[exp]])[0].softmax(dim=-1))
            else:
                if MODE != "finetune_embed":
                    text_feats_batch = torch.nn.functional.normalize(clip_model.encode_text(question[exp_indexer[exp]]))
                else:
                    text_feats_batch = torch.nn.functional.normalize(question[exp_indexer[exp]])
                pred.append((torch.matmul(torch.nn.functional.normalize(torch.unsqueeze(embedding[q_i][0],0)), torch.transpose(text_feats_batch, 0, 1))*model.logit_scale.exp()).softmax(dim=-1))
        if len(pred) == 0:
            continue
        pred_shaped = torch.squeeze(torch.stack(pred), 1)
        pred_argmax = torch.argmax(pred_shaped.detach().cpu(), axis=1).tolist()
        data.append([(g,p) for p,g in zip(pred_argmax,ground_truth.tolist())])
        #print(pred_shaped)
        #print(ground_truth)
        loss = criterion(pred_shaped, ground_truth)
        if model.training and (not optimizer is None):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
        acc_loss += loss.item()
    print_epoch(data,acc_loss,lst, val=False if model.training else True)
    #print(f"Skipped {skipped} games of {len(lst)} this epoch.")
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

    model = Model(seq_model, clip_model=clip).to(DEVICE)

    print(model)
    if train_flag:
        model.train()

    learning_rate = 2e-4
    weight_decay = 1e-3
    num_epochs = 1000#1#

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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
            acc_loss, data = clip_split(model, clip, train, args.experiment, criterion, optimizer)
            print(f"Epoch {epoch} Validation:")
            model.eval()
            #acc_loss, data = do_split(model,val,args.experiment,criterion,global_plan=global_plan, player_plan=player_plan, clip_flag=args.clip)
            acc_loss, data = clip_split(model, clip, val, args.experiment, criterion, optimizer)

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

    model.eval()
    clip_split(model, clip, test, args.experiment, criterion, optimizer)

    print()

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
