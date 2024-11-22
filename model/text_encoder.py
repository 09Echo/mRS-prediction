import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Literal
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertModel(nn.Module):
    def __init__(self,model_path="/biolinkbert",
                 key: Literal['pooler_output', 'last_hidden_state'] = 'pooler_output',):
        super().__init__()

        self.key = key

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        for k in self.model.parameters():
            k.requires_grad = False


    def forward(self, texts):
        encoded_batches = []
        for patient_sentences in texts:
            # 分割句子并编码
            patient_encodings = self.tokenizer(patient_sentences.split(','),max_length=12,return_tensors="pt",padding='max_length',truncation=True)
            patient_encodings = patient_encodings.to("cuda")
            encoded_batches.append(patient_encodings)

        all_patient_sentence_features = []

        for patient_encodings in encoded_batches:
            # 传递给模型并获取特征表示
            outputs = self.model(**patient_encodings)

            if self.key == "pooler_output":
                sentence_features = outputs.pooler_output
            else:
                sentence_features = outputs.last_hidden_state

            # 将句子特征添加到列表中
            all_patient_sentence_features.append(sentence_features)

        # 将所有病人的句子特征堆叠成一个张量
        batch_features = torch.stack(all_patient_sentence_features)  #[B,sentence_num,dim]

        return batch_features

class BertModel_sentence(nn.Module):
    def __init__(self,model_path="/biolinkbert",
                 key: Literal['pooler_output', 'last_hidden_state'] = 'pooler_output',):
        super().__init__()

        self.key = key

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        for k in self.model.parameters():
            k.requires_grad = False

    def forward(self, texts):
        text_inputs = self.tokenizer(texts, max_length=512,return_tensors="pt",padding='max_length',truncation=True)
        text_inputs = text_inputs.to("cuda")

        # with torch.no_grad():
        outputs = self.model(**text_inputs)

        if self.key == "pooler_output":
            text_embeddings = outputs.pooler_output
        else:
            text_embeddings = outputs.last_hidden_state
        return text_embeddings
