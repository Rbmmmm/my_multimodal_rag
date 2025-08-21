import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# 假设 GumbelModalSelector 的定义在这里
# 为了完整性，我根据论文补充一个可能的实现
class GumbelModalSelector(nn.Module):
    def __init__(self, input_dim: int, num_choices: int, hidden_dim: int = 256, tau: float = 1.0, trainable: bool = True):
        super().__init__()
        self.tau = tau
        self.trainable = trainable
        # <<< 改进点 3: 增加模型复杂度，使用 MLP 而非单层线性
        self.selector_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_choices)
        )
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, hard: bool = True):
        logits = self.selector_net(x)
        if not self.training and hard: # 推理时直接用 argmax
            y_onehot = F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.shape[-1]).float()
            return y_onehot, y_onehot, logits

        # 训练时使用 Gumbel-Softmax
        y_onehot = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        probs = F.softmax(logits, dim=-1) # 用于计算 acc 的概率
        return probs, y_onehot, logits

from src.utils.embedding_utils import QueryEmbedder
# from src.models.gumbel_selector import GumbelModalSelector # 使用上面定义的版本

# ========= Config =========
DATA_PATH = "data/ViDoSeek/rag_dataset.json"
SAVE_PATH = "checkpoints/modal_selector_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# <<< 改进点 3: 调整超参数
BATCH_SIZE = 16 # 可以适当增大
EPOCHS = 20 # 增加 Epoch 数量，配合 Early Stopping
LR = 3e-5 # AdamW 更适合配合权重衰减
WEIGHT_DECAY = 0.01
TAU_START = 2.0 # Gumbel 温度初始值
TAU_END = 0.5 # Gumbel 温度最终值

CLASS_NAMES = ["text", "2d_layout", "chart"]
LABEL_ALIAS = {
    "text": "text", "2d_layout": "2d_layout",
    "chart": "chart", "table": "chart",
}

def to_class_id(source_type: str) -> int | None:
    if not isinstance(source_type, str): return None
    st = LABEL_ALIAS.get(source_type.strip().lower())
    if st is None: return None
    return CLASS_NAMES.index(st)

# ========= Dataset =========
# <<< 改进点 4: 优化 Dataset，支持预计算的嵌入
class SelectorDataset(Dataset):
    def __init__(self, data_path: str, embedder: QueryEmbedder, device: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw.get("examples", raw)

        self.embeddings: List[torch.Tensor] = []
        self.labels: List[int] = []
        counts = {k: 0 for k in CLASS_NAMES}
        skipped = 0

        print("Pre-computing embeddings...")
        for ex in tqdm(data):
            q = ex.get("query", "")
            cid = to_class_id(ex.get("meta_info", {}).get("source_type"))
            if cid is not None:
                with torch.no_grad():
                    # 计算嵌入并移动到目标设备
                    emb = embedder.encode(q)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    self.embeddings.append(emb.float().to('cpu')) # 存储在CPU内存中
                self.labels.append(cid)
                counts[CLASS_NAMES[cid]] += 1
            else:
                skipped += 1
        
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.labels)} samples | class counts={counts} | skipped={skipped}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.embeddings[idx], self.labels[idx]

# ========= Train & Eval Loop =========
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, current_tau):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for embs, labels in tqdm(dataloader, desc="Training"):
        embs, labels = embs.to(device), labels.to(device)
        
        model.tau = current_tau # <<< 改进点 3: 更新 Gumbel 温度
        probs, _, logits = model(embs, hard=True)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # <<< 改进点 3: 更新学习率

        total_loss += loss.item() * labels.size(0)
        pred = probs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for embs, labels in tqdm(dataloader, desc="Evaluating"):
            embs, labels = embs.to(device), labels.to(device)
            
            probs, _, logits = model(embs, hard=True)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            pred = probs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train():
    print("🚀 Training Gumbel Selector with supervised labels")

    # 1) 嵌入器
    embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=DEVICE)

    # 2) 数据集和预计算
    full_dataset = SelectorDataset(DATA_PATH, embedder, DEVICE)

    # <<< 改进点 2: 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=full_dataset.labels # 确保分层抽样
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # collate_fn 现在很简单，因为数据已经是 Tensor
    def collate_fn(batch):
        embs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return embs, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3) 模型
    selector = GumbelModalSelector(
        input_dim=embedder.out_dim,
        num_choices=len(CLASS_NAMES),
        tau=TAU_START,
    ).to(DEVICE)
    
    # <<< 改进点 1: 计算类别权重
    class_counts = np.bincount(full_dataset.labels[train_indices])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(DEVICE)
    print(f"Using class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # <<< 改进点 3: 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # <<< 改进点 3: 温度退火
        current_tau = TAU_START - (TAU_START - TAU_END) * (epoch / EPOCHS)
        
        print(f"\n📚 Epoch {epoch}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | Tau: {current_tau:.4f}")
        
        train_loss, train_acc = train_one_epoch(selector, train_loader, optimizer, scheduler, criterion, DEVICE, current_tau)
        print(f"✅ Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = evaluate(selector, val_loader, criterion, DEVICE)
        print(f"📊 Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(selector.state_dict(), SAVE_PATH)
            print(f"💾 New best model saved with Val Acc: {best_val_acc:.2f}% to {SAVE_PATH}")

    print(f"🎉 Training finished. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()