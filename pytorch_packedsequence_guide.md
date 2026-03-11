# PyTorch PackedSequence 原理与使用指南

## 目录
1. [变长数据的问题](#1-变长数据的问题)
2. [PackedSequence 核心原理](#2-packedsequence-核心原理)
3. [核心 API](#3-核心-api)
4. [完整代码示例](#4-完整代码示例)
5. [常见陷阱与解决方案](#5-常见陷阱与解决方案)
6. [总结](#6-总结)

---

## 1. 变长数据的问题

在处理序列数据（如文本、时间序列）时，批次中的序列长度往往不一致：

```
Batch:
  "Hello"      → [H, e, l, l, o]          → 长度 5
  "Hi there"   → [H, i,  , t, h, e, r, e] → 长度 8
  "Yes"        → [Y, e, s]                → 长度 3
```

### 传统做法（Padding）的问题

- **填充到相同长度**：`[H, e, l, l, o, <pad>, <pad>, <pad>]`
- **RNN 仍会处理 padding 位置**，浪费计算资源
- **输出结果包含 padding 的隐藏状态**，干扰后续处理

---

## 2. PackedSequence 核心原理

PackedSequence 是一种**紧凑的存储格式**，让 RNN 跳过 padding 位置，只计算有效数据。

### 直观对比

```
原始数据（带 padding）:
  H e l l o <pad> <pad> <pad>  (长度 5)
  H i _ t h e r e <pad>         (长度 8)
  Y e s <pad> <pad> <pad> <pad> <pad> (长度 3)

PackedSequence 内部结构:
data:    [H, H, Y, e, i, e, l, _, s, l, t, l, h, o, e, <pad>, r, <pad>, e]
         ↑  第1步      ↑  第2步      ↑  第3步       ...按时间步打包

batch_sizes: [3, 3, 3, 2, 2, 2, 2, 1]  ← 每个时间步有多少有效序列
             ↑  t=0时3个序列都有数据
                      ↑  t=3时只剩2个序列
                                ↑  t=7时只剩1个序列

sorted_indices: [1, 0, 2]  ← 按长度降序排序后的原始索引
```

### 执行流程

1. **按长度降序排序**序列
2. **按时间步展开**：t=0 时取所有序列的第1个元素，t=1 时取所有序列的第2个元素...
3. **记录 batch_sizes**：每个时间步有多少序列还有数据
4. RNN 根据 batch_sizes 知道何时停止处理某个序列

---

## 3. 核心 API

| 函数 | 作用 |
|------|------|
| `pack_padded_sequence` | 将 padded 数据打包成 PackedSequence |
| `pad_packed_sequence` | 将 PackedSequence 还原为 padded 张量 |

### 函数签名

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 打包
packed = pack_padded_sequence(
    input,           # 输入张量 [batch, seq_len, features] 或 [batch, seq_len]
    lengths,         # 每个序列的实际长度 [batch]
    batch_first=True,# 输入是否是 batch 优先
    enforce_sorted=False  # 是否要求输入已按长度降序排序
)

# 解包
output, lengths = pad_packed_sequence(
    packed,          # PackedSequence 对象
    batch_first=True,# 输出是否是 batch 优先
    total_length=None  # 指定输出长度（可选）
)
```

---

## 4. 完整代码示例

### 4.1 基础使用示例

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ============ 1. 准备数据 ============
# 假设我们有3个句子，长度分别为5, 3, 4
sentences = [
    [1, 2, 3, 4, 5],      # 长度 5
    [6, 7, 8],             # 长度 3
    [9, 10, 11, 12]        # 长度 4
]

vocab_size = 20
embed_dim = 8
hidden_dim = 16

# ============ 2. Padding ============
max_len = max(len(s) for s in sentences)
padded = torch.zeros(len(sentences), max_len, dtype=torch.long)

for i, sent in enumerate(sentences):
    padded[i, :len(sent)] = torch.tensor(sent)

print("Padded 输入形状:", padded.shape)  # [3, 5]

# ============ 3. 使用 PackedSequence ============

# 步骤1：按长度降序排序（必须的！）
lengths = torch.tensor([len(s) for s in sentences])
lengths_sorted, sort_idx = lengths.sort(descending=True)
padded_sorted = padded[sort_idx]

print(f"\n排序后长度: {lengths_sorted}")  # [5, 4, 3]
print(f"排序索引: {sort_idx}")           # [0, 2, 1]

# 步骤2：打包成 PackedSequence
packed = pack_padded_sequence(
    padded_sorted,
    lengths_sorted,
    batch_first=True,
    enforce_sorted=True
)

print(f"\nPackedSequence:")
print(f"  data 形状: {packed.data.shape}")
print(f"  batch_sizes: {packed.batch_sizes}")
print(f"  sorted_indices: {packed.sorted_indices}")

# ============ 4. 传入 RNN ============

embedding = nn.Embedding(vocab_size, embed_dim)
lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

# 先嵌入，再打包
embedded = embedding(padded_sorted)  # [3, 5, 8]
packed_input = pack_padded_sequence(embedded, lengths_sorted, batch_first=True)

# 传入 LSTM
packed_output, (hidden, cell) = lstm(packed_input)

print(f"\nLSTM 输出:")
print(f"  packed_output.data 形状: {packed_output.data.shape}")

# ============ 5. 解包还原 ============

output, output_lengths = pad_packed_sequence(
    packed_output,
    batch_first=True,
    total_length=max_len
)

print(f"\n解包后输出形状: {output.shape}")  # [3, 5, 16]

# ============ 6. 还原原始顺序 ============

# 如果需要还原原始顺序：
unsort_idx = torch.argsort(sort_idx)
last_hidden_original_order = hidden[-1][unsort_idx]
print(f"\n还原顺序后的隐藏状态: {last_hidden_original_order.shape}")
```

### 4.2 完整分类器示例

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """
    基于 LSTM 的序列分类器，支持变长序列
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 嵌入层（padding_idx=0 表示忽略 padding 位置的梯度）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分类层（双向时隐藏状态维度翻倍）
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        """
        Args:
            x: [batch, seq_len] 输入序列（已按长度降序排序）
            lengths: [batch] 每个序列的实际长度
        
        Returns:
            logits: [batch, num_classes] 分类 logits
        """
        # 1. 嵌入
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # 2. 打包（要求输入已按长度降序排序）
        packed = pack_padded_sequence(
            embedded, lengths,
            batch_first=True,
            enforce_sorted=True
        )
        
        # 3. 过 LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        # hidden: [num_layers * num_directions, batch, hidden_dim]
        
        # 4. 获取最后隐藏状态
        if self.bidirectional:
            # 拼接正向和反向的最后隐藏状态
            # hidden[-2] = 正向最后状态, hidden[-1] = 反向最后状态
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden_cat = hidden[-1]  # [batch, hidden_dim]
        
        hidden_cat = self.dropout(hidden_cat)
        
        # 5. 分类
        logits = self.fc(hidden_cat)  # [batch, num_classes]
        return logits


# ============ 使用示例 ============

# 模拟数据准备函数
def prepare_batch(sentences, pad_value=0):
    """
    将句子列表转换为 padded 张量，并按长度降序排序
    
    Args:
        sentences: 列表的列表，每个内部列表是一个句子（token ID 序列）
        pad_value: padding 值
    
    Returns:
        padded: [batch, max_len] 填充后的张量
        lengths: [batch] 原始长度（已排序）
        sort_idx: [batch] 排序索引（用于还原原始顺序）
    """
    lengths = torch.tensor([len(s) for s in sentences])
    lengths_sorted, sort_idx = lengths.sort(descending=True)
    
    max_len = lengths_sorted[0].item()
    batch_size = len(sentences)
    
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    
    for i, idx in enumerate(sort_idx):
        sent = sentences[idx]
        padded[i, :len(sent)] = torch.tensor(sent)
    
    return padded, lengths_sorted, sort_idx


# 主程序
if __name__ == "__main__":
    # 配置
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    num_classes = 5
    batch_size = 4
    
    # 创建模型
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    )
    
    # 模拟数据（不同长度的序列）
    sentences = [
        [1, 2, 3, 4, 5, 6, 7, 8],      # 长度 8
        [10, 20, 30],                   # 长度 3
        [100, 200, 300, 400, 500],      # 长度 5
        [7, 8, 9, 10, 11, 12]           # 长度 6
    ]
    labels = torch.tensor([0, 1, 2, 3])  # 对应标签
    
    # 准备 batch（自动按长度降序排序）
    padded, lengths, sort_idx = prepare_batch(sentences)
    
    print(f"输入形状: {padded.shape}")  # [4, 8]
    print(f"排序后长度: {lengths}")      # [8, 6, 5, 3]
    
    # 前向传播
    logits = model(padded, lengths)
    print(f"输出形状: {logits.shape}")   # [4, 5]
    
    # 训练步骤
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\n训练步骤完成，损失: {loss.item():.4f}")
    
    # 预测（需要还原原始顺序）
    predictions = logits.argmax(dim=-1)
    unsort_idx = torch.argsort(sort_idx)
    predictions_original = predictions[unsort_idx]
    
    print(f"预测结果: {predictions_original}")
```

### 4.3 DataLoader 集成示例

```python
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, labels, vocab):
        self.texts = [[vocab.get(t, 0) for t in text] for text in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), self.labels[idx]


def collate_fn(batch):
    """
    自定义 collate 函数：处理变长序列
    返回的数据已按长度降序排序，可直接用于 pack_padded_sequence
    """
    sequences, labels = zip(*batch)
    
    # 获取长度
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # 按长度降序排序
    lengths_sorted, sort_idx = lengths.sort(descending=True)
    
    # 排序序列
    sequences_sorted = [sequences[i] for i in sort_idx]
    labels_sorted = torch.tensor([labels[i] for i in sort_idx])
    
    # Padding（使用 pad_sequence 自动处理）
    padded = pad_sequence(sequences_sorted, batch_first=True, padding_value=0)
    
    return padded, lengths_sorted, labels_sorted, sort_idx


# 使用示例
# dataset = TextDataset(texts, labels, vocab)
# dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
#
# for batch_padded, batch_lengths, batch_labels, sort_idx in dataloader:
#     logits = model(batch_padded, batch_lengths)
#     # ... 训练代码
```

---

## 5. 常见陷阱与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `enforce_sorted=True` 报错 | 输入未按长度降序排序 | 手动排序或设为 `False`（有性能损失）|
| 输出长度不匹配 | `total_length` 参数设置不当 | 显式设置 `total_length=max_len` |
| 隐藏状态顺序混乱 | 忘记还原排序 | 使用 `torch.argsort(sort_idx)` 还原 |
| padding_idx 未设置 | 嵌入层处理 padding 位置 | `nn.Embedding(..., padding_idx=0)` |
| 梯度异常 | 对 padding 位置计算了损失 | 使用 mask 忽略 padding 位置 |

### 5.1 重要注意事项

1. **必须按长度降序排序**：`pack_padded_sequence` 要求输入已排序（或设置 `enforce_sorted=False`，但有性能开销）

2. **保留排序索引**：训练时需要保留 `sort_idx` 和 `unsort_idx`，用于还原原始顺序

3. **padding_idx 设置**：`nn.Embedding` 的 `padding_idx` 参数会让 padding 位置的梯度为0，避免干扰训练

4. **双向 LSTM 处理**：双向时需要分别取正向和反向的最后隐藏状态并拼接

---

## 6. 总结

```
┌─────────────────────────────────────────────────────────────┐
│  变长数据处理流程                                            │
├─────────────────────────────────────────────────────────────┤
│  1. 按长度降序排序序列 ← 必须的！                            │
│  2. Padding 到相同长度（为了Embedding能并行处理）            │
│  3. Embedding → PackedSequence (pack_padded_sequence)       │
│  4. 传入 RNN/LSTM/GRU（自动跳过padding位置）                │
│  5. 可选：还原为 padded 格式 (pad_packed_sequence)          │
│  6. 使用 hidden state 做下游任务                            │
└─────────────────────────────────────────────────────────────┘
```

### 核心优势

- **计算高效**：RNN 只处理有效数据，不浪费计算在 padding 上
- **结果准确**：隐藏状态不受 padding 干扰
- **内存优化**：避免存储和传输大量 padding 数据

### 适用场景

- 自然语言处理（不同长度的句子）
- 时间序列分析（不同长度的事件序列）
- 语音处理（不同长度的音频片段）
- 任何需要处理变长序列的 RNN 任务

---

*文档创建时间：2026-03-11*
