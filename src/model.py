import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pinyin
import os
import json
import math
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

INF = 1e5
DEFAULT_BATCH_SIZE = 1000
MAX_WINDOW_SIZE = 200
torch.autograd.set_detect_anomaly(True)
slot_top = {'症状': 11, '检查': 10, '一般信息': 4, '手术': 2}


class RankingLossFunc(nn.Module):
    def __init__(self, mPos, mNeg, gamma):
        '''
        args.mPos = 2.5
        args.mNeg = 0.5
        args.gamma = 0.05
        :param args:
        '''
        super(RankingLossFunc, self).__init__()
        self.mPos = mPos
        self.mNeg = mNeg
        self.gamma = gamma

    def forward(self, logit, target, topk):
       
        val, ind = torch.topk(logit, topk + 1 + 3 + 2, dim=1) 
        logit = F.sigmoid(logit)
        right_logits = torch.where(target == 1, logit, target.float())
        
        right_tag = torch.where(target == 1, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        right_logits = torch.gather(right_logits, dim=1, index=ind)
        right_tag = torch.gather(right_tag, dim=1, index=ind)
        wrong_logits = torch.where(target == 0, logit, torch.Tensor([0]).cuda())
        wrong_tag = torch.where(target == 0, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        wrong_logits = torch.gather(wrong_logits, dim=1, index=ind)
        wrong_tag = torch.gather(wrong_tag, dim=1, index=ind)

       
        data_part1 = torch.exp(self.gamma * (self.mPos - right_logits))
        part1 = right_tag * (torch.log(
            1 + data_part1))  
        data_part2 = torch.exp(self.gamma * (self.mNeg + wrong_logits))
        part2 = wrong_tag * (torch.log(
            1 + data_part2)) 

        loss = torch.sum(part1) + torch.sum(part2) 
        return loss / len(target)


class SelfAtten(nn.Module):
    def __init__(self, input_size, keep_p):
        super(SelfAtten, self).__init__()
        
        self.Linear1 = nn.Linear(input_size, 1)
        self.Dropout = nn.Dropout(1 - keep_p)

    def forward(self, seqs):
        
        a = self.Linear1(seqs)
       
        point = torch.unsqueeze(torch.sum(seqs, -1), -1)
        
        mask = -point.eq(0).float() * INF
        a = a + mask
        p = F.softmax(a, 1)
        
        c = torch.sum(p * seqs, 1)
        c = self.Dropout(c)
        return c



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(2)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, input1, input2, beta=0.5):
        return input1 * beta + (1 - beta) * input2


class Encoder(nn.Module):  
    def __init__(self, input_size, hidden_size, keep_p, add_global=False):
        super(Encoder, self).__init__()
        self.add_global = add_global
        self.Dropout = nn.Dropout(p=1 - keep_p)
        self.BiLSTM = nn.LSTM(input_size=input_size, hidden_size=int(hidden_size / 2), bidirectional=True,
                              batch_first=True)
        self.global_BiLSTM = nn.LSTM(input_size=input_size, hidden_size=int(hidden_size / 2), bidirectional=True,
                                     batch_first=True)
        self.SelfAtten = SelfAtten(input_size=hidden_size, keep_p=keep_p)
        self.globalSelfAtten = SelfAtten(input_size=hidden_size, keep_p=keep_p)
        self.hgate = Gate()
        self.cgate = Gate()

    def forward(self, seqs, seqs_lens):
        

        if self.add_global:
            h_s, _ = self.BiLSTM(seqs)
            h_g, _ = self.global_BiLSTM(seqs)
            h = self.hgate(h_s, h_g)
            c_s = self.SelfAtten(h)
            c_g = self.globalSelfAtten(h)
            c = self.cgate(c_s, c_g)
        else:
            h, _ = self.BiLSTM(seqs)

           
            c = self.SelfAtten(h)

        return h, c


class Attention_Encoder(nn.Module):
    def __init__(self, batch_size, window_size):
        super(Attention_Encoder, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size

    def forward(self, query, keys, values):
        
        slot_value_num = query.shape[0]
        query = query.unsqueeze(0).unsqueeze(0)
        query = query.repeat([keys.shape[0], keys.shape[1], 1, 1])
       
        try:
            p = torch.matmul(query, keys.transpose(2, 3))  
            
        except BaseException:
            print(query.shape)
            print(keys.shape)
        mask = -p.eq(0).float() * INF
        p = F.softmax(p + mask, -1)

        outputs = torch.matmul(p, values)
        
        return outputs


class WLayer(nn.Module):
    def __init__(self, H, W):
        super(WLayer, self).__init__()
       
        w = torch.FloatTensor(torch.rand([1, H, W]))
        
        self.W = nn.Parameter(w, requires_grad=True)

    def forward(self):
        return self.W


class Feedforward_Layer(nn.Module):
    def __init__(self, input_size, num_layers, num_units, outputs_dim, activation, keep_p):
        super(Feedforward_Layer, self).__init__()
        self.Dropout = nn.Dropout(1 - keep_p)
        self.num_layers = num_layers
        self.Activation = activation()
        self.num_units = num_units
        self.input_size = input_size
        self.outputs_dim = outputs_dim
        self.keep_p = keep_p
        if self.num_layers != 0:
            self.layer1 = nn.Linear(input_size, num_units)
            self.layers = nn.ModuleList([nn.Linear(num_units, num_units) for _ in range(num_layers - 1)])
            self.layer2 = nn.Linear(num_units, outputs_dim)
        else:
            self.layer2 = nn.Linear(input_size, outputs_dim)

    def forward(self, inputs):
        if self.num_layers != 0:
            outputs = self.Activation(self.layer1(inputs))
            outputs = self.Dropout(outputs)
            for layer in self.layers:
                outputs = self.Activation(layer(outputs))
                outputs = self.Dropout(outputs)
        else:
            outputs = inputs
        outputs = self.layer2(outputs)
        return outputs


class Model(nn.Module):
    def __init__(self, data, ontology, max_len, window_size, params, batch_size, slots):
        super(Model, self).__init__()
        self.data = data
        self.slots = slots
        self.ontology = ontology
        self.window_size = window_size
        self.max_len = max_len
        self.params = params
        self.batch_size = batch_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.data.dictionary.emb))   

        
        utt_encoder = dict()
        can_encoder = dict()
        w = dict()
        layers = dict()
        select_can_encoder = dict()
        aggregate = dict()
        for slot, _ in ontology.ontology_list:
            aggregate[pinyin.get(slot, format='strip')] = nn.Parameter(
                torch.FloatTensor(torch.rand([self.params['num_units'] * 2, self.params['num_units'] * 2])),
                requires_grad=True)
            select_can_encoder[pinyin.get(slot, format='strip')] = Encoder(self.data.dictionary.emb_size,
                                                                           self.params['num_units'] * 2,
                                                                           self.params['keep_p'],
                                                                           self.params['add_global'])
            utt_encoder[pinyin.get(slot, format='strip')] = Encoder(self.data.dictionary.emb_size,
                                                                    self.params['num_units'], self.params['keep_p'],
                                                                    self.params['add_global'])
            can_encoder[pinyin.get(slot, format='strip')] = Encoder(self.data.dictionary.emb_size,
                                                                    self.params['num_units'], self.params['keep_p'],
                                                                    self.params['add_global'])
            w[pinyin.get(slot, format='strip')] = WLayer(H=self.params['num_units'], W=self.params['num_units'])
           
            layers[pinyin.get(slot, format='strip')] = Feedforward_Layer(2 * self.params['num_units'],
                                                                         self.params['num_layers'],
                                                                         self.params['num_units'], 1, nn.ReLU,
                                                                         self.params['keep_p'])
        self.aggregate = nn.ParameterDict(aggregate)
        self.select_can_encoder = nn.ModuleDict(select_can_encoder)
        self.Utt_Encoder = nn.ModuleDict(utt_encoder)
        self.Can_Encoder = nn.ModuleDict(can_encoder)
        self.W = nn.ModuleDict(w)
        self.FeedforwardLayer = nn.ModuleDict(layers)
        self.Position_Encoding = PositionalEncoding(d_model=self.params['num_units'], dropout=0,
                                                    max_len=MAX_WINDOW_SIZE)
        self.Attention_Encoder = Attention_Encoder(batch_size=batch_size, window_size=window_size)
        self.Sigmoid = F.sigmoid

    def forward(self, windows_utts, windows_utts_lens, labels):
        

        windows_utts = self.embedding(windows_utts)
        self.batch_size = windows_utts.shape[0]
        
        candidate_seqs_dict, candidate_seqs_lens_dict = self.ontology.onto2ids()
        candidate_dict, candidate_dict_lens = self.ontology.candidate2ids()
        for slot in candidate_seqs_dict.keys():
            candidate_seqs_dict[slot] = self.embedding(candidate_seqs_dict[slot])
            if slot == self.ontology.mutual_slot:
                continue
            candidate_dict[slot] = self.embedding(candidate_dict[slot])
            
        utts = torch.reshape(windows_utts, [-1, self.max_len, self.data.dictionary.emb_size])
       
        utts_lens = torch.reshape(windows_utts_lens, [-1])
        
        slot_utt_hs_dict = dict()
        slot_candidate_cs_dict = dict()

        start = 0

        self.slots_train_op = []

        self.slots_pred_logits = []
        self.slots_pred_labels = []
        self.slots_gold_labels = []
        self.slots_loss = []
        for slot in self.slots:
            mask = -torch.unsqueeze(windows_utts_lens, -1).eq(0).float() * INF
            
            _, select_candidate_c = self.select_can_encoder[pinyin.get(slot, format='strip')](candidate_dict[slot],
                                                                                              candidate_dict_lens[slot])
           
            utt_h, _ = self.Utt_Encoder[pinyin.get(slot, format='strip')](utts, utts_lens)
           
            utt_h = torch.reshape(utt_h, [-1, self.window_size, self.max_len, self.params['num_units']])

            
            _, candidate_c = self.Can_Encoder[pinyin.get(slot, format='strip')](candidate_seqs_dict[slot],
                                                                                candidate_seqs_lens_dict[slot])

            
            slot_utt_hs_dict[slot] = {"utt_h": utt_h}
            slot_candidate_cs_dict[slot] = {"candidate_c": candidate_c}
            utt_h, _ = self.Utt_Encoder[pinyin.get(self.ontology.mutual_slot, format='strip')](utts, utts_lens)
            utt_h = torch.reshape(utt_h, [-1, self.window_size, self.max_len, self.params['num_units']])
            

            _, candidate_c = self.Can_Encoder[pinyin.get(self.ontology.mutual_slot, format='strip')](
                candidate_seqs_dict[self.ontology.mutual_slot],
                candidate_seqs_lens_dict[self.ontology.mutual_slot])
            

            slot_utt_hs_dict[slot]["status_utt_h"] = utt_h
            slot_candidate_cs_dict[slot]["status_candidate_c"] = candidate_c
            status_utt_h = slot_utt_hs_dict[slot]["status_utt_h"]
            status_candidate_c = slot_candidate_cs_dict[slot]["status_candidate_c"]
            q_status = self.Position_Encoding(self.Attention_Encoder(status_candidate_c, status_utt_h, status_utt_h))
            

            slot_utt_h = slot_utt_hs_dict[slot]["utt_h"]
            
            slot_candidate_c = slot_candidate_cs_dict[slot]["candidate_c"]
            

            q_slot = self.Position_Encoding(self.Attention_Encoder(slot_candidate_c, slot_utt_h, slot_utt_h))
            

            slot_value_num = slot_candidate_c.shape[0]
            status_num = status_candidate_c.shape[0]

            
            w = self.W[pinyin.get(slot, format='strip')]().repeat([self.batch_size, 1, 1])

            
            co = torch.reshape(
                torch.reshape(
                    torch.matmul(
                        torch.matmul(
                            torch.reshape(
                                q_slot,
                                [self.batch_size, self.window_size * slot_value_num, self.params['num_units']]
                            ),
                            w
                        ),
                        torch.reshape(
                            q_status,
                            [self.batch_size, self.window_size * status_num, self.params['num_units']]
                        ).transpose(1, 2),

                    )  
                    ,
                    [self.batch_size, self.window_size, slot_value_num, self.window_size * status_num]
                ),
                [self.batch_size, self.window_size, slot_value_num, self.window_size, status_num]
            )
            co_mask = -co.eq(0).float() * INF
            p = co + co_mask

            
            p = F.softmax(p, 3)  
            
            q_status_slot = q_status.unsqueeze(-1) \
                .unsqueeze(-1). \
                repeat([1, 1, 1, 1, self.window_size, slot_value_num]). \
                permute([0, 4, 5, 1, 2, 3])
            

            q_status_slot = torch.sum(p.unsqueeze(-1) * q_status_slot, 3)
           
            q_slot = q_slot.unsqueeze(3).repeat([1, 1, 1, status_num, 1])
           
            features = torch.cat([q_slot, q_status_slot], -1)
            
            weight_matrix = self.aggregate[pinyin.get(slot, format='strip')].repeat([self.batch_size, 1, 1])
            
            features_weight = torch.reshape(torch.sum(torch.reshape(
                torch.matmul(
                    torch.matmul(
                        torch.reshape(
                            features,
                            [self.batch_size, self.window_size * slot_value_num * status_num,
                             2 * self.params['num_units']]
                        ),
                        weight_matrix
                    ),

                    select_candidate_c.transpose(-1, -2)
                    

                ),  
                [self.batch_size, self.window_size, slot_value_num * status_num, slot_value_num * status_num]
            ) * torch.eye(slot_value_num * status_num).long().cuda(), dim=-1, keepdim=True
                                                      ),
                                            [self.batch_size, self.window_size, slot_value_num, status_num, 1]
                                            )
            

            features_weight_mask = -features_weight.eq(0).float() * INF
            mask = mask.unsqueeze(2).unsqueeze(3).repeat([1, 1, slot_value_num, status_num, 1])
            
            features_p = features_weight + features_weight_mask + mask
            features_p = F.softmax(features_p, 1)
           

            features = torch.sum(features_p * features, 1)
            
            logits = torch.sum(torch.matmul(torch.reshape(features, [self.batch_size, slot_value_num * status_num, -1]),
                                       select_candidate_c.transpose(-1, -2)) * torch.eye(
                    slot_value_num * status_num).long().cuda(), dim=-1, keepdim=True)

            
            logits = torch.reshape(logits, [-1, slot_value_num * status_num])
            
            slot_pred_logits = logits
            

            
            slot_pred_labels = torch.tensor(slot_pred_logits > 0, dtype=torch.float32)

            
            slot_gold_labels = labels[:, start: start + slot_value_num * status_num]

            
            start += slot_value_num * status_num

            yield slot, slot_pred_logits, slot_gold_labels, slot_pred_labels


class MIE:
    def __init__(self, data, ontology, **kw):

        self.mPos = kw["params"]["mPos"]
        self.mNeg = kw["params"]["mNeg"]
        self.gamma = kw["params"]["gamma"]
        self.slot_top = slot_top
        self.data = data
        self.ontology = ontology
        self.slots = [item[0] for item in self.ontology.ontology_list if item[0] != self.ontology.mutual_slot]
        
        self.max_len = self.data.max_len  
        self.infos = dict()
        
        if 'location' in kw:
            self._load(kw['location'])
        
        elif 'params' in kw:
            self.params = kw['params']
            
        for dataset in ('train', 'dev'):
            self.infos[dataset] = dict()
            for slot in self.slots:
                self.infos[dataset][slot] = {
                    'ps': [],
                    'rs': [],
                    'f1s': [],
                    'losses': []
                }
            self.infos[dataset]['global'] = {
                'ps': [],
                'rs': [],
                'f1s': [],
                'losses': []
            }

    def compute_loss(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_loss = [[] for i in range(len(self.slots_loss))]
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            windows_utts_batch, windows_utts_lens_batch, labels_batch = torch.from_numpy(
                windows_utts_batch).long().cuda(), torch.from_numpy(
                windows_utts_lens_batch).long().cuda(), torch.from_numpy(labels_batch).long().cuda()
            true_batch_size = windows_utts_batch.shape[0]
            window_size = windows_utts_batch.shape[1]
            self.model.eval()
            self.model.batch_size = true_batch_size
            slots_pred_logits_batch = []
            slots_glod_labels_batch = []
            with torch.no_grad():
                for slot, slot_pred_logits, slot_gold_labels, slot_pred_labels in self.model(windows_utts_batch,
                                                                                             windows_utts_lens_batch,
                                                                                             labels_batch): 
                    slots_pred_logits_batch.append(slot_pred_logits)
                    slots_glod_labels_batch.append(slot_gold_labels)
            losses = dict([(slot, None) for slot in self.slots])
            losses['global'] = None
            criterion = nn.BCEWithLogitsLoss()
            for i in range(len(slots_pred_logits_batch)):
                loss = criterion(slots_pred_logits_batch[i], slots_glod_labels_batch[i].float())
                slot = self.slots[i]
                losses[slot] = loss
                loss = loss.cpu().detach().numpy()
                slots_loss[i].append(loss)
        losses['global'] = float(np.mean(np.concatenate(slots_loss, -1)))

        return losses

    def inference_with_score(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_pred_labels = [[] for i in range(len(self.slots))]
        slots_gold_labels = [[] for i in range(len(self.slots))]
        slots_logits_labels = [[] for i in range(len(self.slots))]
       
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
           
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            windows_utts_batch, windows_utts_lens_batch, labels_batch = torch.from_numpy(
                windows_utts_batch).long().cuda(), torch.from_numpy(
                windows_utts_lens_batch).long().cuda(), torch.from_numpy(labels_batch).long().cuda()
            true_batch_size = windows_utts_batch.shape[0]
            window_size = windows_utts_batch.shape[1]
            self.model.batch_size = true_batch_size
            self.model.eval()
            slots_pred_labels_batch = []
            slots_logits_labels_batch = []
            with torch.no_grad():
               
                for slot, slot_pred_logits, slot_gold_labels, slot_pred_labels in self.model(windows_utts_batch,
                                                                                             windows_utts_lens_batch,
                                                                                             labels_batch):
                    

                    slots_pred_labels_batch.append(slot_pred_labels.cpu().detach().numpy())
                    slots_logits_labels_batch.append(slot_pred_logits.cpu().detach().numpy())
            start = 0
            labels_batch = labels_batch.cpu().numpy()
            for i, slot_pred_labels_batch in enumerate(slots_pred_labels_batch):
                end = start + slot_pred_labels_batch.shape[1]
                slots_gold_labels[i].append(labels_batch[:, start: end])
                slots_pred_labels[i].append(slot_pred_labels_batch)
                slots_logits_labels[i].append(slots_logits_labels_batch[i])
                start = end
        
        for i in range(len(slots_gold_labels)):
            slots_gold_labels[i] = np.concatenate(slots_gold_labels[i], 0)
            slots_pred_labels[i] = np.concatenate(slots_pred_labels[i], 0)
            slots_logits_labels[i] = np.concatenate(slots_logits_labels[i], 0)
        
        return slots_logits_labels, slots_pred_labels, slots_gold_labels

    def inference(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_pred_labels = [[] for i in range(len(self.slots))]
        slots_gold_labels = [[] for i in range(len(self.slots))]
        
        for i, batch in enumerate(self.data.batch(name, batch_size, False)):
            
            if (i + 1) * batch_size > num:
                break
            windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
            windows_utts_batch, windows_utts_lens_batch, labels_batch = torch.from_numpy(
                windows_utts_batch).long().cuda(), torch.from_numpy(
                windows_utts_lens_batch).long().cuda(), torch.from_numpy(labels_batch).long().cuda()
            true_batch_size = windows_utts_batch.shape[0]
            window_size = windows_utts_batch.shape[1]
            self.model.batch_size = true_batch_size
            self.model.eval()
            slots_pred_labels_batch = []
            with torch.no_grad():
                
                for slot, slot_pred_logits, slot_gold_labels, slot_pred_labels in self.model(windows_utts_batch,
                                                                                             windows_utts_lens_batch,
                                                                                             labels_batch):
                    

                    slots_pred_labels_batch.append(slot_pred_labels.cpu().detach().numpy())
            start = 0
            labels_batch = labels_batch.cpu().numpy()
            for i, slot_pred_labels_batch in enumerate(slots_pred_labels_batch):
                end = start + slot_pred_labels_batch.shape[1]
                slots_gold_labels[i].append(labels_batch[:, start: end])
                slots_pred_labels[i].append(slot_pred_labels_batch)
                start = end
       
        for i in range(len(slots_gold_labels)):
            slots_gold_labels[i] = np.concatenate(slots_gold_labels[i], 0)
            slots_pred_labels[i] = np.concatenate(slots_pred_labels[i], 0)
       
        return slots_pred_labels, slots_gold_labels

    def _evaluate(self, pred_labels, gold_labels):
        def _add_ex_col(x):
            col = 1 - np.sum(x, -1).astype(np.bool).astype(np.float32)
            col = np.expand_dims(col, -1)
            x = np.concatenate([x, col], -1)
            return x

        pred_labels = _add_ex_col(pred_labels)
        gold_labels = _add_ex_col(gold_labels)
        tp = np.sum((pred_labels == gold_labels).astype(np.float32) * pred_labels, -1)
        pred_pos_num = np.sum(pred_labels, -1)
        gold_pos_num = np.sum(gold_labels, -1)
        p = (tp / pred_pos_num)
        r = (tp / gold_pos_num)
        p_add_r = p + r
        p_add_r = p_add_r + (p_add_r == 0).astype(np.float32)
        f1 = 2 * p * r / p_add_r

        return p, r, f1

    def _load(self, batch_size, location, filename):
        self.model = Model(self.data, self.ontology, self.max_len, 5, self.params, batch_size, self.slots)
        self.model.load_state_dict(torch.load(os.path.join(location, filename)))

    def evaluate(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        slots_pred_labels, slots_gold_labels = \
            self.inference(name, num, batch_size)
        info = dict()
        for slot in self.slots:
            info[slot] = {
                'p': None,
                'r': None,
                'f1': None
            }
        info['global'] = {
            'p': None,
            'r': None,
            'f1': None
        }
        for i, (slot_pred_labels, slot_gold_labels) in \
                enumerate(zip(slots_pred_labels, slots_gold_labels)):
            p, r, f1 = map(
                lambda x: float(np.mean(x)),
                self._evaluate(slot_pred_labels, slot_gold_labels)
            )
            slot = self.slots[i]
            info[slot]['p'] = p
            info[slot]['r'] = r
            info[slot]['f1'] = f1

        pred_labels = np.concatenate(slots_pred_labels, -1)
        gold_labels = np.concatenate(slots_gold_labels, -1)
        p, r, f1 = map(
            lambda x: float(np.mean(x)),
            self._evaluate(pred_labels, gold_labels)
        )
        info['global']['p'] = p
        info['global']['r'] = r
        info['global']['f1'] = f1

        return info

    def _add_infos(self, name, info):
        for slot in info.keys():
            if isinstance(info[slot], float):
                
                self.infos[name][slot]['losses'].append(info[slot])
            elif isinstance(info[slot], dict):
                
                for key in info[slot].keys():
                    self.infos[name][slot][key + 's'].append(info[slot][key])

    def save(self, location, filename, save_graph=True):
        if not os.path.exists(location):
            os.makedirs(location)
        torch.save(self.model.state_dict(), os.path.join(location, filename))

    def train(self,
              epoch_num,
              batch_size,
              tbatch_size,
              start_lr,
              end_lr,
              alpha,
              cuda,
              location=None,
              filename=None
              ):
        
        decay = (end_lr / start_lr) ** (1 / epoch_num)
        lr = start_lr
        self.model = Model(self.data, self.ontology, self.max_len, 5, self.params, batch_size, self.slots).cuda()
        

        loader = data.DataLoader(self.data, batch_size=batch_size, shuffle=True, drop_last=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion_rank = RankingLossFunc(self.mPos, self.mNeg, self.gamma)
        for i in range(epoch_num):
            self.model.train()
            pbar = tqdm(loader, desc='Epoch {}'.format(i + 1))
            for batch in pbar:
                '''if flag == 0:
                    flag += 1
                else :
                    break'''
                self.slots_pred_logits = []
                self.slots_pred_labels = []
                self.slots_gold_labels = []
                windows_utts, windows_utts_len, labels = batch

                
                windows_utts, windows_utts_len, labels = windows_utts.long().cuda(), windows_utts_len.long().cuda(), labels.long().cuda()
                

                self.slots_loss = []

                for slot, slot_pred_logits, slot_gold_labels, slot_pred_labels in self.model(windows_utts,
                                                                                             windows_utts_len,
                                                                                             labels):
                    try:
                        loss = criterion(slot_pred_logits, slot_gold_labels.float())
                    except Exception:
                        print(slot_pred_logits.shape, slot_gold_labels.shape)
                    loss_rank = criterion_rank(slot_pred_logits, slot_gold_labels, self.slot_top[slot])
                    
                    optimizer.zero_grad()
                    
                    Loss = loss + loss_rank
                    

                    try:
                        Loss.backward()
                    except BaseException:
                        print("Loss.backward() Exception :")
                        for name, parms in self.model.named_parameters():
                            print(slot)
                            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                                  ' -->grad_value:', parms.grad)
                            assert (False)

                    optimizer.step()
                    self.slots_pred_logits.append(slot_pred_logits)
                    self.slots_pred_labels.append(slot_pred_labels)
                    self.slots_gold_labels.append(slot_gold_labels)

                    self.slots_loss.append(loss)

            train_prf = self.evaluate('train', tbatch_size, tbatch_size)
            train_loss = self.compute_loss('train', tbatch_size, tbatch_size)
            dev_prf = self.evaluate('dev', batch_size=tbatch_size)
            dev_loss = self.compute_loss('dev', batch_size=tbatch_size)

            self._add_infos('train', train_prf)
            self._add_infos('train', train_loss)
            self._add_infos('dev', dev_prf)
            self._add_infos('dev', dev_loss)

            
            print('''Epoch {}: train_loss={:.4}, dev_loss={:.4}
                                                train_p={:.4}, train_r={:.4}, train_f1={:.4}
                                                dev_p={:.4}, dev_r={:.4}, dev_f1={:.4}'''.
                  format(i + 1, train_loss['global'], dev_loss['global'],
                         train_prf['global']['p'], train_prf['global']['r'],
                         train_prf['global']['f1'], dev_prf['global']['p'],
                         dev_prf['global']['r'], dev_prf['global']['f1']))

            if len(self.infos['dev']['global']['f1s']) > 0 and location:
                if dev_prf['global']['f1'] >= max(self.infos['dev']['global']['f1s']):
                    test_prf = self.evaluate('test', batch_size=tbatch_size)
                    self.save(location, filename)
                    print('保存在{}！'.format(location))

            try:
                print('Now test result: f1={:.4}'.format(test_prf['global']['f1']))
            except NameError:
                pass








