import sys, torch, random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from x_transformers import Decoder, Encoder, TransformerWrapper
from numpy.core.fromnumeric import reshape
import torch.nn as nn, numpy as np

random.seed(0)
torch.manual_seed(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

class Model(nn.Module):
    def __init__(self, seq_model_type=0):
        super(Model, self).__init__()
        self.seq_model_type = seq_model_type

        my_rnn = lambda i,o: nn.GRU(i,o)
        #my_rnn = lambda i,o: nn.LSTM(i,o)

        plan_emb_in = 81
        plan_emb_out = 32
        q_emb = 100

        self.plan_embedder0 = my_rnn(plan_emb_in,plan_emb_out)
        self.plan_embedder1 = my_rnn(plan_emb_in,plan_emb_out)
        self.plan_embedder2 = my_rnn(plan_emb_in,plan_emb_out)

        # self.dialogue_listener = my_rnn(1126,768)
        dlist_hidden = 1024
        frame_emb = 512
        text_emb = 1024 + 2
        # TODO: Change this back for BERT
        # drnn_in = 1024 + 2 + q_emb + frame_emb
        # drnn_in = 1024 + 2
        drnn_in = 512 + 2 + q_emb + frame_emb

        # my_rnn = lambda i,o: nn.GRU(i,o)
        #my_rnn = lambda i,o: nn.LSTM(i,o)

        if seq_model_type==0:
            self.dialogue_listener_rnn = nn.GRU(drnn_in,dlist_hidden)
            self.dialogue_listener = lambda x: \
                self.dialogue_listener_rnn(x.reshape(-1,1,drnn_in))[0]
        elif seq_model_type==1:
            self.dialogue_listener_rnn = nn.LSTM(drnn_in,dlist_hidden)
            self.dialogue_listener = lambda x: \
                self.dialogue_listener_rnn(x.reshape(-1,1,drnn_in))[0]
        elif seq_model_type==2:
            mask_fun = lambda x: torch.triu(torch.ones(x.shape[0],x.shape[0]),diagonal=1).bool().to(DEVICE)
            sincos_fun = lambda x:torch.transpose(torch.stack([
                torch.sin(2*np.pi*torch.tensor(list(range(x)))/x),
                torch.cos(2*np.pi*torch.tensor(list(range(x)))/x)
                ]),0,1).reshape(-1,1,2)
            self.dialogue_listener_lin1 = nn.Linear(drnn_in,dlist_hidden-2)
            self.dialogue_listener_attn = nn.MultiheadAttention(dlist_hidden, 8)
            self.dialogue_listener_wrap = lambda x: self.dialogue_listener_attn(x,x,x,attn_mask=mask_fun(x))
            self.dialogue_listener = lambda x: self.dialogue_listener_wrap(torch.cat([
                sincos_fun(x.shape[0]).float().to(DEVICE),
                self.dialogue_listener_lin1(x).reshape(-1,1,dlist_hidden-2)
            ], axis=-1))[0]
        elif seq_model_type==3:
            print("Using Decoder based Transformer")
            self.dialogue_listener = Decoder(
                    dim = 1126,
                    depth = 1,
                    heads = 1,
                    attn_num_mem_kv = 16, # 16 memory key / values
                    attn_talking_heads = True,  # turn on information exchange between attention heads
                    ff_glu=True,
                    rel_pos_bias = True,  # adds relative positional bias to all attention layers, a la T5
                    rotary_pos_emb = True,
                    layer_dropout = 0.2,   # stochastic depth - dropout entire layer
                    attn_dropout = 0.2,    # dropout post-attention
                    ff_dropout = 0.2       # feedforward dropout
            ).to(DEVICE)
            self.final_drop = nn.Dropout(0.2).to(DEVICE)
            self.final_ff = nn.Linear(1126, 1024).to(DEVICE)
        else:
            print('Sequence model type must be in (0: GRU, 1: LSTM, 2: Transformer), but got ', seq_model_type)
            exit()

        conv_block = lambda i,o,k,p,s: nn.Sequential(
            nn.Conv2d(   i,   o, k, padding=p, stride=s),
            nn.BatchNorm2d(o),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(o),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        '''
        self.conv = nn.Sequential(
            conv_block(   3,   8, 3, 1, 1),
            # conv_block(   3,   8, 5, 2, 2),
            conv_block(   8,  32, 5, 2, 2),
            conv_block(  32, frame_emb//4, 5, 2, 2),
            nn.Conv2d( frame_emb//4, frame_emb, 3),nn.ReLU(),
        )
        '''
        self.conv = None

        qlayer = lambda i,o : nn.Sequential(
            nn.Linear(i,512),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512,o),
            nn.Dropout(0.2),
            # nn.Softmax(-1)
        )

        q_in_size = 3*plan_emb_out+dlist_hidden+q_emb

        self.q01 = qlayer(q_in_size,2)
        self.q02 = qlayer(q_in_size,2)
        self.q03 = qlayer(q_in_size,2)

        self.q11 = qlayer(q_in_size,3)
        self.q12 = qlayer(q_in_size,3)
        self.q13 = qlayer(q_in_size,22)

        self.q21 = qlayer(q_in_size,3)
        self.q22 = qlayer(q_in_size,3)
        self.q23 = qlayer(q_in_size,22)

    def forward(self,game,global_plan=False, player_plan=False, clip=True):
        retval = []

        l = list(game)
        _,d,q,f,c_f = zip(*list(game))

        h = None
        f=c_f
        assert f==c_f

        # TODO: Change this into an actual fix, the last frame should not have inverted dimensions???
        #f[-1] = f[-1] if f[-1].shape[0]==512 else f[-1].T
        f = np.array(f, dtype=np.float64)
        # f = torch.tensor(f).permute(0,3,1,2).float().to(DEVICE)
        # flt_lst = [(a,b) for a,b in zip(d,q) if (not a is None) or (not b is None)]
        # if not flt_lst:
        #     return []
        # d,q = zip(*flt_lst)
        text_emb_size = 514 if clip else 1024
        d = [np.concatenate(([int(x[0][1]==2),int(x[0][1]==1)],x[0][-1])) if not x is None else np.zeros(text_emb_size) for x in d]
        def parse_q(q):
            if not q is None:
                q ,l = q
                q = np.concatenate([
                    onehot(q[2],2),
                    onehot(q[3],2),
                    onehot(q[4][0][0]+1,2),
                    onehot(game.materials_dict[q[4][0][1]],len(game.materials_dict)),
                    onehot(q[4][1][0]+1,2),
                    onehot(game.materials_dict[q[4][1][1]],len(game.materials_dict)),
                    onehot(q[4][2]+1,2),
                    onehot(q[5][0][0]+1,2),
                    onehot(game.materials_dict[q[5][0][1]],len(game.materials_dict)),
                    onehot(q[5][1][0]+1,2),
                    onehot(game.materials_dict[q[5][1][1]],len(game.materials_dict)),
                    onehot(q[5][2]+1,2)
                    ])
            else:
                q = np.zeros(100)
                l = None
            return q, l
        try:
            sel1 = int([x[0][2] for x in q if not x is None][0] == 1)
            sel2 = 1 - sel1
        except Exception as e:
            sel1 = 0
            sel2 = 0

        def parse_q_contr(q):
            if not q is None:
                q ,l = q
                q =
            else:
                q = np.zeros(512)
                l = None
            return q, l
        try:
            sel1 = int([x[0][2] for x in q if not x is None][0] == 1)
            sel2 = 1 - sel1
        except Exception as e:
            sel1 = 0
            sel2 = 0
        q = [parse_q(x) for x in q]
        q, l = zip(*q)


        if not global_plan and not player_plan:
            plan_emb = torch.cat([
                self.plan_embedder0(torch.stack(list(map(torch.tensor,game.global_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                self.plan_embedder1(torch.stack(list(map(torch.tensor,game.player1_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                self.plan_embedder2(torch.stack(list(map(torch.tensor,game.player2_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0]
            ])
            plan_emb = 0*plan_emb
        elif global_plan:
            plan_emb = torch.cat([
                self.plan_embedder0(torch.stack(list(map(torch.tensor,game.global_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                self.plan_embedder1(torch.stack(list(map(torch.tensor,game.player1_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                self.plan_embedder2(torch.stack(list(map(torch.tensor,game.player2_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0]
            ])
        else:
            plan_emb = torch.cat([
                0*self.plan_embedder0(torch.stack(list(map(torch.tensor,game.global_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                sel1*self.plan_embedder1(torch.stack(list(map(torch.tensor,game.player1_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0],
                sel2*self.plan_embedder2(torch.stack(list(map(torch.tensor,game.player2_plan))).reshape(-1,1,81).float().to(DEVICE))[0][-1][0]
            ])

        u = torch.cat((
            torch.tensor(d).float().to(DEVICE),
            torch.tensor(q).float().to(DEVICE),
            # FIX THIS TO OG!
            #torch.tensor(f).float().to(DEVICE) if clip else
            torch.tensor(np.squeeze(f,axis=1) if len(f.shape)==3 else f).float().to(DEVICE) if clip else
            self.conv(torch.tensor(f).permute(0,2,1).float().to(DEVICE)).reshape(-1,512), # Use MineCLIP directly if possible
            ),axis=-1)

        if self.seq_model_type==3:
            u = torch.unsqueeze(u, 0)
        # Original
        u = u.float().to(DEVICE)

        y = self.dialogue_listener(u)
        y = y.reshape(-1,y.shape[-1])
        if self.seq_model_type==3:
            y = self.final_drop(y)
            y = self.final_ff(y)
            y = y.reshape(-1,y.shape[-1])
        if all([x is None for x in l]):
            return []

        fun_lst = [self.q01,self.q02,self.q03,self.q11,self.q12,self.q13,self.q21,self.q22,self.q23]
        fun = lambda x: [f(x) for f in fun_lst]

        retval = [(_l,fun(torch.cat((plan_emb,torch.tensor(_q).float().to(DEVICE),_y)))) for _y, _q, _l in zip(y,q,l)if not _l is None]
        return retval
