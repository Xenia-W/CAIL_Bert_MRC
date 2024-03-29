import torch
import torch.nn as nn
from config import args
from util.nn import LSTM
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class QaExtract(BertPreTrainedModel):
    def __init__(self, config):
        super(QaExtract, self).__init__(config)
        self.bert = BertModel(config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False

        self.start_lstm = LSTM(input_size=config.hidden_size,
                               hidden_size=args.LSTM_hidden_size,
                               bidirectional=True,
                               batch_first=True,
                               dropout=args.LSTM_dropout)
        self.start_linear = nn.Linear(args.LSTM_hidden_size * 2, 1)

        self.end_lstm = LSTM(input_size=config.hidden_size + args.LSTM_hidden_size * 2,
                             hidden_size=args.LSTM_hidden_size,
                             bidirectional=True,
                             batch_first=True,
                             dropout=args.LSTM_dropout)
        self.end_linear = nn.Linear(args.LSTM_hidden_size * 2, 1)

        self.apply(self.init_bert_weights)

        self.type_lstm = LSTM(input_size=config.hidden_size + args.LSTM_hidden_size * 2,
                              hidden_size=args.LSTM_hidden_size,
                              bidirectional=True,
                              batch_first=True,
                              dropout=args.LSTM_dropout)
        self.type_linear = nn.Linear(args.LSTM_hidden_size * 2, 4)

        # for domain type task
        self.bert_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.domain_type_classifier = nn.Linear(config.hidden_size, 2)

        '''basic
        self.classifier = nn.Linear(config.hidden_size, 2)
        '''

        '''trivial
        self.classifier_start = nn.Linear(config.hidden_size, 1)
        self.classifier_end = nn.Linear(config.hidden_size,1)
        self.apply(self.init_bert_weights)
        self.answer_type_classifier = nn.Linear(config.hidden_size, 4)
        '''

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)  # (B, T, 768)
        seq_len = sequence_output.size(1)
        seq_len = torch.Tensor([seq_len for _ in range(sequence_output.size(0))]).cuda()

        '''prediciton of baseline
        logits = self.classifier(sequence_output)                                          # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)                                 # ((B, T, 1), (B, T, 1))
        start_logits = start_logits.squeeze(-1)                                            # (B, T)
        end_logits = end_logits.squeeze(-1)                                                # (B, T)
        '''

        '''prediction of trivial improve
        start_logits = self.classifier_start(sequence_output)
        end_logits = self.classifier_end(start_logits*sequence_output)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        answer_type_logits = self.answer_type_classifier(pooled_output)
        '''

        start_input = sequence_output
        start_output = self.start_lstm((start_input, seq_len))[0]
        start_logits = self.start_linear(start_output).squeeze(-1)

        end_input = torch.cat([sequence_output, start_output], dim=-1)
        end_output = self.end_lstm((end_input, seq_len))[0]
        end_logits = self.end_linear(end_output).squeeze(-1)

        type_input = torch.cat([sequence_output, end_output], dim=-1)
        type_output = torch.max(self.type_lstm((type_input, seq_len))[0], dim=1)[0]
        type_logits = type_output
        answer_type_logits = self.type_linear(type_logits).squeeze(-1)

        # for domain type task
        last_sep = sequence_output[:, -1]
        sep_output = self.activation(self.bert_dense(last_sep))
        domain_type_logits = self.domain_type_classifier(sep_output)

        return start_logits, end_logits, answer_type_logits, domain_type_logits
