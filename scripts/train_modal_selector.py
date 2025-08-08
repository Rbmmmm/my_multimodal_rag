# File: my_multimodal_rag/scripts/train_modal_selector.py

import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.gumbel_selector import GumbelModalSelector
from src.utils.embedding_utils import QueryEmbedder

class SelectorDS(Dataset):
    def __init__(self, queries, labels, embedder: QueryEmbedder):
        self.embedder = embedder
        self.X = embedder.encode(queries)          # [N, D]
        self.y = torch.tensor(labels, dtype=torch.long, device=self.X.device)

    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_labels(path: str):
    queries, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            queries.append(row["query"])
            labels.append(int(row["label"]))
    return queries, labels

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_path = "data/ViDoSeek/selector_labels.csv"
    queries, labels = load_labels(labels_path)

    embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)  # 以后替换 Qwen3
    ds = SelectorDS(queries, labels, embedder)

    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = GumbelModalSelector(input_dim=embedder.out_dim, hidden_dim=256, num_choices=2, tau=1.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_acc, best_path = 0.0, Path("checkpoints/modal_selector_text_image.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(10):
        model.train()
        # 训练时用 logits 直接做 CE（最稳）
        for X, y in train_dl:
            _, logits, _ = model(X, training=True, hard=False)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 退火
        model.set_temperature(max(0.3, model.tau * 0.9))

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                probs, logits, choice = model.infer(X)
                pred = choice
                correct += (pred == y).sum().item()
                total   += y.numel()
        acc = correct / max(1, total)
        print(f"Epoch {epoch:02d} | val_acc={acc:.4f} | tau={model.tau:.2f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved to {best_path}")

    print(f"Best val_acc={best_acc:.4f}, weights={best_path}")

if __name__ == "__main__":
    main()