import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# ===============================================================
# 1. 从我们分离好的文件中导入核心模块
# ===============================================================
# 假设您的目录结构是:
# src/
# ├── models/
# │   └── gumbel_selector.py
# ├── training/
# │   └── train_gumbel.py
# └── utils/
#     └── embedding_utils.py
from src.models.gumbel_selector import GumbelModalSelector
from src.utils.embedding_utils import QueryEmbedder

# ===============================================================
# 2. 数据集类 (与之前相同，保持不变)
# ===============================================================
class SelectorDataset(Dataset):
    """
    为模态选择器准备数据集。
    - 从 JSON 文件加载查询和标签。
    - 使用提供的 embedder 预先计算所有查询的嵌入。
    """
    def __init__(self, data_path: str, embedder: QueryEmbedder, class_map: Dict[str, str], class_names: List[str]):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw.get("examples", raw)

        self.embeddings: List[torch.Tensor] = []
        self.labels: List[int] = []
        counts = {k: 0 for k in class_names}
        skipped = 0

        print(" Pre-computing embeddings for the dataset...")
        for ex in tqdm(data, desc=" Pre-computing Embeddings"):
            q = ex.get("query", "")
            source_type = ex.get("meta_info", {}).get("source_type")
            
            # 使用 class_map 将原始标签映射到目标类别
            st = class_map.get(source_type.strip().lower()) if isinstance(source_type, str) else None
            cid = class_names.index(st) if st in class_names else None

            if cid is not None:
                with torch.no_grad():
                    emb = embedder.encode(q)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    self.embeddings.append(emb.float().cpu()) # 存储在CPU，送入GPU前再转移
                self.labels.append(cid)
                counts[class_names[cid]] += 1
            else:
                skipped += 1
        
        self.labels = np.array(self.labels)
        print(f" Loaded {len(self.labels)} samples | Class counts={counts} | Skipped={skipped}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.embeddings[idx], self.labels[idx]

# ===============================================================
# 3. 训练与评估循环 (与之前相同，保持不变)
# ===============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, current_tau):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for embs, labels in tqdm(dataloader, desc="Training"):
        embs, labels = embs.to(device), labels.to(device)
        model.tau = current_tau
        probs, _, logits = model(embs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * embs.size(0)
        pred = probs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for embs, labels in tqdm(dataloader, desc="Evaluating"):
            embs, labels = embs.to(device), labels.to(device)
            probs, _, logits = model(embs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * embs.size(0)
            pred = probs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

# ===============================================================
# 4. 主训练函数
# ===============================================================
def main(args):
    print("🚀 Training Gumbel Selector with supervised labels")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # --- 类别定义 ---
    CLASS_NAMES = ["text", "2d_layout", "chart"]
    LABEL_ALIAS = {
        "text": "text", "2d_layout": "2d_layout",
        "chart": "chart", "table": "chart",
    }
    
    # --- 嵌入器 ---
    print(f" [1/5] Loading embedding model: {args.embedder_name}")
    embedder = QueryEmbedder(model_name=args.embedder_name, device=device)

    # --- 数据集 ---
    print(f" [2/5] Loading and processing dataset from {args.data_path}")
    full_dataset = SelectorDataset(args.data_path, embedder, LABEL_ALIAS, CLASS_NAMES)
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=full_dataset.labels
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    def collate_fn(batch):
        embs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return embs, labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    print(f" Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- 模型 ---
    print(" [3/5] Initializing the Gumbel Modal Selector model")
    selector = GumbelModalSelector(
        input_dim=embedder.out_dim,
        num_choices=len(CLASS_NAMES),
        hidden_dim=args.hidden_dim,
        tau=args.tau_start,
    ).to(device)

    # --- 损失函数、优化器、调度器 ---
    print(" [4/5] Setting up loss, optimizer, and schedulers")
    class_counts = np.bincount(full_dataset.labels[train_indices])
    class_weights = (1. / torch.tensor(class_counts, dtype=torch.float32)).to(device)
    class_weights /= class_weights.sum()
    print(f" Using class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)

    # --- 训练循环 ---
    print(" [5/5] Starting the training loop...")
    best_val_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        current_tau = args.tau_start - (args.tau_start - args.tau_end) * (epoch / args.epochs)
        
        print(f"\n📚 Epoch {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Tau: {current_tau:.4f}")
        
        train_loss, train_acc = train_one_epoch(selector, train_loader, optimizer, scheduler, criterion, device, current_tau)
        print(f" ✅ Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = evaluate(selector, val_loader, criterion, device)
        print(f" 📊 Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(selector.state_dict(), args.save_path)
            print(f" 💾 New best model saved with Val Acc: {best_val_acc:.2f}% to {args.save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f" Patience: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"🛑 Early stopping triggered after {epoch} epochs.")
            break

    print(f"\n🎉 Training finished. Best Val Acc: {best_val_acc:.2f}% saved to {args.save_path}")

# ===============================================================
# 5. 命令行参数解析
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Gumbel Modal Selector.")
    
    # --- 文件路径 ---
    parser.add_argument('--data_path', type=str, default="data/ViDoSeek/rag_dataset.json", help="Path to the training data.")
    parser.add_argument('--save_path', type=str, default="checkpoints/modal_selector_best.pt", help="Path to save the best model.")
    parser.add_argument('--embedder_name', type=str, default="BAAI/bge-m3", help="Name of the sentence-transformer model for embeddings.")
    
    # --- 模型超参数 ---
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension of the selector's MLP.")
    
    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    
    # --- Gumbel-Softmax 超参数 ---
    parser.add_argument('--tau_start', type=float, default=2.0, help="Initial temperature for Gumbel-Softmax.")
    parser.add_argument('--tau_end', type=float, default=0.5, help="Final temperature for Gumbel-Softmax.")
    
    args = parser.parse_args()
    main(args)