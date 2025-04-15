# import pandas as pd
# import torch
# import numpy as np
# from transformers import MobileBertForSequenceClassification,MobileBertTokenizer
# from tqdm import tqdm
#
# GPU = torch.cuda.is_available()
# device = torch.device("cuda" if GPU else "cpu")
# print("Using device: ", device)
#
# data_path = "val_data.csv"
# df = pd.read_csv(data_path, encoding="utf-8")
#
#
#
#
#
# data_X = list(df['content'].values)
# labels = df['score'].values
#
#
#
# tokenizers = MobileBertTokenizer.from_pretrained('mobilebert_uncased', do_lower_case=True)
# inputs = tokenizers(data_X,truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
#
# batch_size = 8
#
#
# test_inputs = torch.tensor(input_ids)
# test_labels = torch.tensor(labels)
# test_masks = torch.tensor(attention_mask)
# test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
# test_sampler = torch.utils.data.RandomSampler(test_data)
# test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
#
# model = MobileBertForSequenceClassification.from_pretrained("mobilebert_custom_model_imdb.pt")
# model.to(device)
#
# model.eval()
#
# test_pred = []
# test_true = []
#
# for batch in tqdm(test_dataloader):
#     batch_ids, batch_mask, batch_labels = batch
#
#     batch_ids = batch_ids.to(device)
#     batch_mask = batch_mask.to(device)
#     batch_labels = batch_labels.to(device)
#
#     with torch.no_grad():
#         output = model(batch_ids, attention_mask=batch_mask)
#
#     logits = output.logits
#     pred = torch.argmax(logits, dim=1)
#     test_pred.extend(pred.cpu().numpy())
#     test_true.extend(batch_labels.cpu().numpy())
#
# test_accuracy = np.sum(np.array(test_pred) == np.array(test_true))/len(test_pred)
# print(test_accuracy)
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
# # 예측값과 실제값을 numpy 배열로 변환
# test_pred = np.array(test_pred)
# test_true = np.array(test_true)
#
# # 혼동 행렬 계산
# cm = confusion_matrix(test_true, test_pred)
#
# # 그래프 그리기
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

import torch
import pandas as pd
import numpy as np
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 디바이스 설정
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

# 랜덤 시드 고정 (재현성)
torch.manual_seed(42)
np.random.seed(42)

# 데이터 불러오기
data_path = "val_data.csv"
df = pd.read_csv(data_path, encoding="utf-8")

# 결측값 처리 + 문자열 변환

data_X = [str(x) for x in df["content"].values]
labels = df["score"].values

# 토크나이징
tokenizers = MobileBertTokenizer.from_pretrained('mobilebert_uncased', do_lower_case=True)
inputs = tokenizers(
    data_X,
    truncation=True,
    max_length=256,
    add_special_tokens=True,
    padding="max_length"
)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# TensorDataset 준비
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 모델 로드 (폴더 경로 사용)
model_path = "mobilebert_custom_model_imdb.pt"  #
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 테스트
test_pred = []
test_true = []

for batch in tqdm(test_dataloader, desc="Testing"):
    batch_ids, batch_mask, batch_labels = batch
    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)

    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 정확도 계산
test_accuracy = np.mean(np.array(test_pred) == np.array(test_true))
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# ---------------- 시계열 분석 ----------------


import numpy as np
from collections import Counter

# 예측 결과가 test_pred에 들어 있다고 가정
total = len(test_pred)
count = Counter(test_pred)
pos = count[1]
neg = count[0]


import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import MobileBertTokenizer

# 데이터 준비
data_X = [str(x) for x in df["content"].values]
labels = df["score"].values

# MobileBERT tokenizer 로드
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

# 텍스트 토크나이징
inputs = tokenizer(
    data_X,
    truncation=True,
    max_length=256,
    add_special_tokens=True,
    padding="max_length",
    return_tensors="pt"  # 바로 텐서로 반환
)

# 입력 텐서와 라벨 텐서 준비
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.tensor(labels)

# Dataset 생성
dataset = TensorDataset(input_ids, attention_mask, labels)

# DataLoader 생성
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)






print(f"전체 리뷰 {total:,}건 중:\n")
print(f"긍정 예측: 약 {pos:,}건 ({pos / total:.1%})")
print(f"부정 예측: 약 {neg:,}건 ({neg / total:.1%})")

