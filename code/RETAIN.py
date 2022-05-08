#!/usr/bin/env python

import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

script_dir = os.path.dirname(__file__)
DATA_PATH = "../data"
DATA_PATH = os.path.join(script_dir, DATA_PATH)

class DataProcessor:

    def __init__(self):
        self.__pids = pickle.load(open(os.path.join(DATA_PATH,'full/pids.pkl'), 'rb'))
        self.__hfs = pickle.load(open(os.path.join(DATA_PATH,'full/hfs.pkl'), 'rb'))
        self.__seqs = pickle.load(open(os.path.join(DATA_PATH,'full/seqs.pkl'), 'rb'))
        self.__types = pickle.load(open(os.path.join(DATA_PATH,'full/types.pkl'), 'rb'))

        self.num_codes = len(self.__types)

    from torch.utils.data import Dataset
    class CustomDataset(Dataset):        
        def __init__(self, seqs, hfs, pids):
            self.__x = seqs
            self.__y = hfs
            self.__p = pids
        
        def __len__(self):        
            return len(self.__x)
        
        def __getitem__(self, index):
            pid = self.__p[index]
            return self.__x[pid], self.__y[pid]     

    def __collate_fn(self, data):
        sequences, labels = zip(*data)
        y = torch.tensor(labels, dtype=torch.float)
        
        num_patients = len(sequences)
        num_visits = [len(patient) for patient in sequences]
        num_codes = [len(visit) for patient in sequences for visit in patient]

        max_num_visits = max(num_visits)
        max_num_codes = max(num_codes)
        
        x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
        rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
        masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
        rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
        for i_patient, patient in enumerate(sequences):
            for j_visit, visit in enumerate(patient):
                v_l = len(visit)
                p_l = len(patient)
                x[i_patient, j_visit, :v_l] = torch.tensor(visit, dtype=torch.long)
                rev_x[i_patient, p_l - j_visit - 1, :v_l] = torch.tensor(visit, dtype=torch.long)
                masks[i_patient, j_visit, :v_l].fill_(1)
                rev_masks[i_patient, p_l - j_visit - 1, :v_l].fill_(1)
        return x, masks, rev_x, rev_masks, y
    
    def getDataLoaders(self, batch_size):
        from torch.utils.data.dataset import random_split
        from torch.utils.data import DataLoader

        dataset = self.CustomDataset(self.__seqs, self.__hfs, self.__pids)
        split = int(len(dataset)*0.8)
        lengths = [split, len(dataset) - split]
        train_dataset, val_dataset = random_split(dataset, lengths)

        def load_data(train_dataset, val_dataset, collate_fn):            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
            return train_loader, val_loader

        train_loader, val_loader = load_data(train_dataset, val_dataset, self.__collate_fn)
        return train_loader, val_loader    

class RETAIN(nn.Module):
    
    def __init__(self, num_codes, embedding_dim=128, embeddings=None):
        super().__init__()
        self.__embedding_dim=embedding_dim
        if embeddings is not None:
            self.__embedding = nn.Embedding.from_pretrained(embeddings=embeddings)
            (_, self.__embedding_dim) = embeddings.shape
        else:
            self.__embedding = nn.Embedding(num_codes, self.__embedding_dim)
        self.__rnn_a = nn.GRU(self.__embedding_dim, self.__embedding_dim, batch_first=True)
        self.__rnn_b = nn.GRU(self.__embedding_dim, self.__embedding_dim, batch_first=True)
        self.__att_a = self.AlphaAttention(self.__embedding_dim)        
        self.__att_b = self.BetaAttention(self.__embedding_dim)
        self.__fc = nn.Linear(self.__embedding_dim, 1)
        self.__sigmoid = nn.Sigmoid()
    
    def forward(self, x, masks, rev_x, rev_masks):
        rev_x = self.__embedding(rev_x).float()
        rev_x = self.__sum_embeddings_with_mask(rev_x, rev_masks)
        g, _ = self.__rnn_a(rev_x)
        h, _ = self.__rnn_b(rev_x)
        alpha = self.__att_a(g)
        beta = self.__att_b(h)
        c = self.__attention_sum(alpha, beta, rev_x, rev_masks)
        logits = self.__fc(c)
        probs = self.__sigmoid(logits)
        return probs.squeeze()

    class AlphaAttention(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.__a_att = nn.Linear(hidden_dim, 1)

        def forward(self, g):
            g = self.__a_att(g)
            g = torch.softmax(g, 1)
            return g

    class BetaAttention(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.__b_att = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h):
            h = self.__b_att(h)
            h = torch.tanh(h)
            return h

    def __attention_sum(self, alpha, beta, rev_v, rev_masks):
        m = torch.sum(rev_masks,2).unsqueeze(2)    
        s = torch.sum(alpha * beta * rev_v * ( m > 0).type(torch.float), dim=1)
        return s

    def __sum_embeddings_with_mask(self, x, masks):
        x = x * masks.unsqueeze(-1)
        x = torch.sum(x, dim = -2)
        return x
    
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class ModelProcessor:
    def __init__(self, n_epochs, batch_size, vectors=None):
        self.__n_epochs = n_epochs
        self.__embeddings = vectors
        self.__batch_size = batch_size

    def __eval(self):
        self.__model.eval()
        y_pred = torch.LongTensor()
        y_score = torch.Tensor()
        y_true = torch.LongTensor()
        self.__model.eval()
        for x, masks, rev_x, rev_masks, y in self.__val_loader:
            y_logit = self.__model(x, masks, rev_x, rev_masks)
            y_hat = y_logit >= 0.5 
            y_score = torch.cat((y_score,  y_logit.detach().to('cpu')), dim=0)
            y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        roc_auc = roc_auc_score(y_true, y_score)
        return p, r, f, roc_auc

    def __train(self):
        for epoch in range(self.__n_epochs):
            self.__model.train()
            train_loss = 0
            for x, masks, rev_x, rev_masks, y in self.__train_loader:
                self.__optimizer.zero_grad()
                y_hat = self.__model(x, masks, rev_x, rev_masks)
                loss = self.__criterion(y_hat, y)
                loss.backward()
                self.__optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(self.__train_loader)
            print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
            p, r, f, roc_auc =self.__eval()
            print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch+1, p, r, f, roc_auc))
        return round(roc_auc, 2)

    def train(self):
        dataProcessor = DataProcessor()
        self.__train_loader, self.__val_loader = dataProcessor.getDataLoaders(self.__batch_size)
        self.__model = RETAIN(num_codes=dataProcessor.num_codes, embeddings=self.__embeddings)
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=1e-3)
        self.__criterion = nn.BCELoss()

        self.__train()

if __name__ == '__main__':
    modelProcessor = ModelProcessor(n_epochs=5, batch_size=100)
    modelProcessor.train()
    print('RETAIN model training complete!')