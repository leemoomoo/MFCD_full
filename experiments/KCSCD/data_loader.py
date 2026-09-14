import json
import torch

class DataLoader(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.ptr = 0
        self.data = []
        data_file = args.data_dir+args.train_set
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.knowledge_dim = int(args.n_skill)
    def next_batch(self, is_train=True):
        if self.is_end():
            return None, None, None, None
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, affects = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                # skill_ids.append(knowledge_code)
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            # if(is_train):
            #    affects.append(log['affect'])
            input_knowledge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.tensor(affects)


    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class ValTestDataLoader(object):
    def __init__(self, args, type):
        self.ptr = 0
        self.data = []
        if type == 'val':
            data_file = args.data_dir+args.val_set
        elif type == 'test':
            data_file = args.data_dir+args.test_set
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
            self.knowledge_dim = int(args.n_skill)

    def next_batch(self, is_train=True):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, affects = [], [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                # skill_ids.append(knowledge_code)
                knowledge_emb[knowledge_code - 1] = 1.0
            # if (is_train):
                # affects.append(log['affect'])
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.tensor(affects)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
