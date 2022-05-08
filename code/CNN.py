import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch.optim as optim
import string
import pickle

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

vec_filename = 'icd_code_embeddings.vec'
script_dir = os.path.dirname(__file__)

DATA_PATH = "../data"
DATA_PATH = os.path.join(script_dir, DATA_PATH)

class DataProcessor:
    def __getDataAndLabels(self):
        codes = pickle.load(open(os.path.join(DATA_PATH,'full/codes.pkl'), 'rb'))
        codes_data = codes.values()        
        self.num_classes = len(codes_data)
        return codes_data, np.eye(self.num_classes)

    def __tokenizeData(self, data):
        from nltk.tokenize import word_tokenize
    
        index = 0
        sequences = []
        self.token2idx = defaultdict(int)
        self.m_length_toks = 0
        
        self.token2idx['<end>'] = index
        index += 1
        for d in data:
            t_data = word_tokenize(d.translate(str.maketrans('','',string.punctuation)))
            sequences.append(t_data)
            for t_d in t_data:
                if t_d not in self.token2idx:
                    self.token2idx[t_d] = index
                    index += 1
            self.m_length_toks = max(self.m_length_toks, len(t_data))
        self.token2idx['<unk>'] = index

        def enc_sequences():
            enc_sequences = []
            for seq in sequences:
                p_seq = seq + ['<end>'] * (self.m_length_toks - len(seq))
                enc_sequences.append([self.token2idx.get(s) if s in self.token2idx else self.token2idx.get('<unk>') for s in p_seq])
            return np.array(enc_sequences)
        
        return enc_sequences()

    def __generateEmbeddings(self):
        from tqdm import tqdm_notebook
        with open(os.path.join(os.path.join(script_dir, '../data'),  vec_filename), 'r', encoding='utf-8', errors='ignore', newline='\n') as vec_file:
            _, dim = map(int, vec_file.readline().split())
            embs = np.random.uniform(0.1, .80, (len(self.token2idx), dim))      
            for l in tqdm_notebook(vec_file):
                toks = l.rstrip().split(' ')
                if toks[0] in self.token2idx:
                    embs[self.token2idx[toks[0]]] = np.array(toks[1:], dtype=np.float)
            embs[self.token2idx['<end>']] = np.zeros((dim,))
            return embs

    def getDataLoaders(self, batch_size=32):        
        from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

        self.data, self.labels = self.__getDataAndLabels()        
        encoded_data = self.__tokenizeData(data=self.data)
        self.embeddings = torch.tensor(self.__generateEmbeddings())

        def splitData():
            from sklearn.model_selection import train_test_split
            return train_test_split(encoded_data, self.labels, test_size=0.2)

        X_train, X_test, y_train, y_test = splitData()
        tr_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
        te_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float))
        full_data = TensorDataset(torch.tensor(encoded_data, dtype=torch.long), torch.tensor(self.labels, dtype=torch.float))
        return  DataLoader(dataset=tr_data, sampler=RandomSampler(tr_data), batch_size=batch_size), \
                DataLoader(dataset=te_data, sampler=SequentialSampler(te_data), batch_size=batch_size), \
                DataLoader(dataset=full_data, sampler=SequentialSampler(full_data), batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self, num_classes, embeddings=None, kernel_sizes=[2,3,4], num_filters=[100,100,100], dropout=0.5):
        super(CNN, self).__init__()
        self.__embedding_dim=128
        if embeddings is not None:
            self.__embeddings = nn.Embedding.from_pretrained(embeddings=embeddings)
            (_, self.__embedding_dim) = embeddings.shape
        else:
            self.__embeddings = nn.Embedding(num_classes, self.__embedding_dim)
        self.__conv_list = nn.ModuleList([nn.Conv1d(in_channels=self.__embedding_dim, 
                                                  out_channels=num_filters[i], 
                                                  kernel_size=kernel_sizes[i]) for i in range(len(kernel_sizes))])
        self.__fc = nn.Linear(np.sum(num_filters), num_classes)
        self.__dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = self.__embeddings(data).float()
        x = x.permute(0, 2, 1)
        res_conv_lst = [F.relu(conv(x)) for conv in self.__conv_list]
        mp_lst = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in res_conv_lst]
        x = torch.cat([x_p.squeeze(dim=2) for x_p in mp_lst], dim=1)
        x = self.__dropout(x)
        x = self.__fc(x)
        return x

class ModelProcessor:
    def __init__(self, n_epochs, batch_size):
        self.__n_epochs = n_epochs
        self.__batch_size = batch_size

    def __train(self):    
        for epoch in range(self.__n_epochs):
            self.__model.train()
            train_loss = 0
            for sequences, labels in self.__train_loader:
                self.__optimizer.zero_grad()                       
                y_hat = self.__model.forward(sequences)
                loss = self.__criterion(y_hat, labels)
                train_loss += loss.item()
                loss.backward()
                self.__optimizer.step()
            train_loss = train_loss / len(self.__train_loader)
            print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
            p, r, f = self.__eval()
            print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}'.format(epoch+1, p, r, f))

    def __eval(self):
        from sklearn.metrics import precision_recall_fscore_support
        self.__model.eval()
        accuracy = []
        y_pred = torch.LongTensor()
        y_true = torch.LongTensor()
        for sequences, labels in self.__test_loader:
            y_hat = self.__model.forward(sequences)
            accuracy.append(self.__get_accuracy(y_hat, labels))
            y_hat = y_hat >= 0.5
            y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
            y_true = torch.cat((y_true, labels.detach().to('cpu')), dim=0)        
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print('Validation accuracy: {:.2f}'.format(np.mean(accuracy)))
        return p, r, f

    def __get_accuracy(self, y_hat, label):        
        y_hat_classes = y_hat.argmax(dim=-1)
        lab = label.argmax(dim=-1)
        correct_pred = y_hat_classes.eq(lab).sum()
        accuracy = correct_pred / self.__batch_size
        return accuracy.detach().numpy()

    def get_vectors(self, dim=0):        
        self.__model.eval()
        emb_data = []
        labels = []
        for sequences, lab in self.__full_loader:
            cur_embd = self.__model.forward(sequences)
            emb_data += list(cur_embd.detach().numpy())
            labels += list(lab.numpy())
        dim = len(emb_data[0]) if dim == 0 else min(dim, len(emb_data[0]))
        embs = np.random.uniform(0.10, 0.80, (len(emb_data), dim))
        for idx, l in enumerate(emb_data):            
            embs[idx] = np.array(l[:dim], dtype=np.float)

        return torch.tensor(np.array(embs)), torch.tensor(np.array(labels))

    def train(self):
        self.__dataProcessor = DataProcessor()
        self.__train_loader, self.__test_loader, self.__full_loader = self.__dataProcessor.getDataLoaders(self.__batch_size)
        self.__model = CNN(embeddings=self.__dataProcessor.embeddings, num_classes=self.__dataProcessor.num_classes)
        self.__optimizer = optim.Adadelta(self.__model.parameters(), lr=0.001)
        self.__criterion = nn.CrossEntropyLoss()
 
        self.__train()

    def predict(self, condition):       
        from nltk.tokenize import word_tokenize 
        t_data = word_tokenize(condition.translate(str.maketrans('','',string.punctuation)))
        p_seq = t_data + ['<end>'] * (self.__dataProcessor.m_length_toks - len(t_data))
        tokenized_data = [self.__dataProcessor.token2idx.get(s) if s in self.__dataProcessor.token2idx 
                            else self.__dataProcessor.token2idx.get('<unk>') for s in p_seq]
        data = torch.tensor(tokenized_data).unsqueeze(0)
        y_hat = self.__model.forward(data)
        y_hat = y_hat >= 0.5
        y_hat = (y_hat == True).nonzero(as_tuple=True)
        print ("Result -  {}".format(y_hat))

if __name__ == '__main__':
    modelProcessor = ModelProcessor(n_epochs=5, batch_size=100)
    modelProcessor.train()

    print('CNN model training complete!')
    
    
    

