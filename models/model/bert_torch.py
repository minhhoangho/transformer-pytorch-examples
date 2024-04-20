import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
import torch
import torch.nn as nn
from torchcrf import CRF
class BertCRF(nn.Module):
    def __init__(self, max_len, output_dim, n_tags):
        self.n_tags = n_tags # N tag for tagging
        super(BertCRF, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, output_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*output_dim, n_tags)
        self.crf = CRF(n_tags, batch_first=True)
        self.max_len = max_len


    def forward(self, input_ids, attention_mask, token_type_ids):
        # Truncate/pad input sequences
        if input_ids.size(1) > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            token_type_ids = token_type_ids[:, :self.max_len]
            attention_mask = attention_mask[:, :self.max_len]
        elif input_ids.size(1) < self.max_len:
            padding = torch.zeros((input_ids.size(0), self.max_len - input_ids.size(1)), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
            token_type_ids = torch.cat([token_type_ids, padding], dim=1)
            attention_mask = torch.cat([attention_mask, padding], dim=1)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output[0]
        lstm_output, _ = self.lstm(sequence_output)
        output = self.fc(lstm_output)
        return self.crf.decode(output), -self.crf(output)


    def get_loss(self, x, tags, mask=None):
        if mask is None:
            return -self.crf(x, tags)
        else:
            return -self.crf(x, tags, mask=mask, reduction='mean')

    def get_accuracy(self, x, tags, mask=None):
        predicted_tags = self.crf.decode(x)
        correct_predictions = (tags == predicted_tags).float()
        if mask is not None:
            correct_predictions *= mask
        accuracy = correct_predictions.sum() / correct_predictions.numel()
        return accuracy