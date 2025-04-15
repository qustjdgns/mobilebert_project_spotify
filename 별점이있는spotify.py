import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 디바이스 설정
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

# 1. 학습 시 경고 메세지 제거
logging.set_verbosity_error()

# 2. 데이터 불러오기 및 확인
path = "train_data.csv"
df = pd.read_csv(path, encoding="utf-8")



# X, y 데이터
data_X = [str(x) for x in df["content"].values]  # 문자열로 변환
labels = df["score"].values

print("### 데이터 샘플 ###")
print("리뷰 문장 : ", data_X[:5])
print("긍정/부정 : ", labels[:5])

# 3. 토크나이저 로드 및 토큰화
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

print("\n###토큰화 결과 샘플###")
for j in range(3):
    print(f"\n{j + 1}번째 데이터")
    print("데이터 : ", data_X[j])
    print("토큰 : ", input_ids[j])
    print("어텐션 마스크 :", attention_mask[j])

# 4. 학습/검증 데이터셋 분리
train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, validation_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 5. 데이터셋 및 DataLoader 구성
batch_size = 8

train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_mask)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_mask)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 6. 모델 및 옵티마이저 설정
model = MobileBertForSequenceClassification.from_pretrained('mobilebert_uncased', num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * epochs
)

# 7. 학습 및 검증 루프
epoch_results = []

for e in range(epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}", leave=True)

    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # 학습 정확도
    model.eval()
    train_pred, train_true = [], []
    for batch in tqdm(train_dataloader, desc=f"Evaluation Train Epoch {e + 1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)

        train_pred.extend(pred.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = np.sum(np.array(train_pred) == np.array(train_true)) / len(train_pred)

    # 검증 정확도
    val_pred, val_true = [], []
    for batch in tqdm(validation_dataloader, desc=f"Evaluation Validation Epoch {e + 1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)

        val_pred.extend(pred.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())

    val_accuracy = np.sum(np.array(val_pred) == np.array(val_true)) / len(val_pred)
    epoch_results.append((avg_train_loss, train_accuracy, val_accuracy))

# 8. 결과 출력
for idx, (loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(f"Epoch {idx}: Train Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
# 10. 그래프 시각화
epochs_range = range(1, len(epoch_results) + 1)
train_loss = [x[0] for x in epoch_results]
train_acc = [x[1] for x in epoch_results]
val_acc = [x[2] for x in epoch_results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1. Training Loss
axes[0].plot(epochs_range, train_loss, marker='o', color='blue', label='Train Loss')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)
axes[0].legend()

# 2. Training & Validation Accuracy
axes[1].plot(epochs_range, train_acc, marker='o', color='green', label='Train Accuracy')
axes[1].plot(epochs_range, val_acc, marker='o', color='red', label='Validation Accuracy')
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# 9. 모델 저장
print("\n### 모델 저장 ###")
save_path = "mobilebert_custom_model_imdb"
model.save_pretrained(save_path + ".pt")
print("모델 저장 완료")

