# import torch
# import torch.nn as nn
# import transformers
# from transformers import BertModel
# from transformers.models.bert.tokenization_bert import BertTokenizer
#
#
# base_model_name = "cointegrated/rubert-tiny"
# base_rubert = BertModel.from_pretrained(base_model_name)
# tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)
#
# class EthicsSentBERT(nn.Module):
#     def __init__(self,
#                  base_model: BertModel,
#                  dropout: float = 0.3,
#                  last_embedding_dim: int = 312,
#                  classification_dim: int = 3):
#         super(EthicsSentBERT, self).__init__()
#
#         self.bert_model = base_model
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(last_embedding_dim, classification_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, ids, mask, token_type_ids):
#         _, pooled_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
#
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.linear(dropout_output)
#         final_layer = self.relu(linear_output)
#
#         return final_layer
#
#
# # !g1.1
# class Predictor:
#     def __init__(self, trained_model, max_length=MAX_LENGTH):
#         self.tokenizer = tokenizer
#         self.model = trained_model
#         self.max_length = max_length
#
#     def predict(self, sentence, proba: bool = False):
#         inputs = self.tokenizer.encode_plus(
#             sentence,
#             None,
#             pad_to_max_length=True,
#             add_special_tokens=True,
#             return_attention_mask=True,
#             max_length=self.max_length,
#         )
#
#         ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
#         token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0)
#         mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)
#
#         with torch.no_grad():
#             logits = self.model(
#                 ids=ids,
#                 mask=mask,
#                 token_type_ids=token_type_ids
#             )
#
#         res = logits.argmax(dim=1)
#
#         if proba:
#             res = nn.functional.softmax(logits, dim=-1).numpy()[0]
#
#         return res
#
#
# def only_char_left(text):
#     text_new = ''
#     for i in text:
#         if i.isalpha() or i == ' ':
#             text_new += i
#     return text_new
#
#
# model_trained = EthicsSentBERT(base_model=base_rubert)
# checkpoint = torch.load("./state_dicts/best_checkpoint.pth")
# model_trained.load_state_dict(checkpoint)
#
# predictor = Predictor(model_trained.cpu())


def predict(sentence, ml_algorithm):
    return {'is_two_categories': True,
            'is_percents': True,
            'category_1': 'Communication',
            'category_2': 'Price',
            'sentiment': '+',
            'category_1_percent': 0.93,
            'category_2_percent': 0.87,
            'sentiment_percent': 0.94}
