import torch
import torch.nn as nn
import torch.nn.functional as F
import config.args as args

from util.nn import LSTM, Linear
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

class QaExtract(BertPreTrainedModel):
    def __init__(self, config):
        super(QaExtract, self).__init__(config)
        # 1. fixed bert
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
        # for p in self.bert.parameters():
        #     p.requires_grad = False

        # 2. Attention Flow Layer
        self.att_weight_c = Linear(config.hidden_size, 1)
        self.att_weight_q = Linear(config.hidden_size, 1)
        self.att_weight_cq = Linear(config.hidden_size, 1)

        # 3. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=config.hidden_size * 4,
                                   hidden_size=args.LSTM_hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.LSTM_dropout)

        self.modeling_LSTM2 = LSTM(input_size=args.LSTM_hidden_size * 2,
                                   hidden_size=args.LSTM_hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.LSTM_dropout)
        # 4. Output layer
        self.p1_weight_g = Linear(config.hidden_size * 4, 1, dropout=args.LSTM_dropout)
        self.p1_weight_m = Linear(args.LSTM_hidden_size * 2, 1, dropout=args.LSTM_dropout)
        self.p2_weight_g = Linear(config.hidden_size * 4, 1, dropout=args.LSTM_dropout)
        self.p2_weight_m = Linear(args.LSTM_hidden_size * 2, 1, dropout=args.LSTM_dropout)

        self.output_LSTM = LSTM(input_size=args.LSTM_hidden_size * 2,
                                hidden_size=args.LSTM_hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.LSTM_dropout)

        self.dropout = nn.Dropout(p=args.LSTM_dropout)

        '''
        # self.classifier = nn.Linear(config.hidden_size, 2)
        self.classifier_start = nn.Linear(config.hidden_size, 1)
        self.classifier_end = nn.Linear(config.LSTM_hidden_size,1)        
        '''

        self.answer_type_classifier = nn.Linear(config.hidden_size, 4)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze(-1)
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq


            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            print("intput: q:{} c:{} cq:{} a:{} c2q:{} b:{}".format(q.shape,c.shape,cq.shape,a.shape,c2q_att.shape,b.shape))
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            print("input: c_len:{} q2c:{}".format(c_len,q2c_att.shape))
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze(-1)
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze(-1)

            return p1, p2

        def split_qc_embedding(inputs, segment_ids):
            #
            max_index = 0
            idxs = []
            for i in range(segment_ids.size(0)):
                index = 0
                while index < segment_ids.size(1) and segment_ids[i][index] == 0:
                    index += 1
                idxs.append(index)
                max_index = max(max_index,index)

            q,c = inputs.split([max_index,segment_ids.size(1)-max_index],dim=1)
            return q, c

        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)  # (B, T, 768)
        # context_embedding : [B, c_len, hidden_size]
        query_embedding,context_embedding = split_qc_embedding(sequence_output,token_type_ids)
        c_len = context_embedding.size(1)
        q_len = query_embedding.size(1)
        c_lens = torch.Tensor([c_len for _ in range(sequence_output.size(0))]).cuda()
        print("---------------------processing----------------------------")
        print("c:{} q:{} c_lens:{}".format(context_embedding.shape,query_embedding.shape,c_lens.shape))
        print("c_len:{} q_len{}".format(c_len,q_len))

        # 2. attention flow layer    g : [B,c_len,hidden_size*4]
        g = att_flow_layer(context_embedding,query_embedding)

        # 3. modeling layer         m : [B,c_len,LSTM_hidden_size * 2]
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]

        print("g:{} m:{}".format(g.shape,m.shape))

        # 4. output layer           p1 : [B,c_len]
        p1, p2 = output_layer(g, m, c_lens)


        '''
        # logits = self.classifier(sequence_output)                                          # (B, T, 2)
        # start_logits, end_logits = logits.split(1, dim=-1)                                 # ((B, T, 1), (B, T, 1))
        # start_logits = start_logits.squeeze(-1)                                            # (B, T)
        # end_logits = end_logits.squeeze(-1)                                                # (B, T)
        start_logits = self.classifier_start(sequence_output)
        end_logits = self.classifier_end(start_logits * sequence_output)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        '''
        start_logits = F.pad(p1,pad=(q_len,0,0,0),mode="constant",value=0)
        end_logits = F.pad(p2,pad=(q_len,0,0,0),mode="constant",value=0)
        answer_type_logits = self.answer_type_classifier(pooled_output)

        return start_logits, end_logits, answer_type_logits