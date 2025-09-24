# 词汇表扩展使用指南

## 概述

本功能允许您扩增原始模型的词汇表大小，同时保留已有的embeddings。这在LFQ（Learned Feature Quantization）场景中特别有用，因为您需要为新的量化token添加词汇表条目。

## 主要功能

### 1. 词汇表扩展
- 自动计算新的词汇表大小
- 保留所有已有的embeddings
- 为新token初始化embeddings
- 同时扩展input和output embeddings
- 支持冻结原始embeddings，只训练新扩展部分

### 2. LFQ集成
- 自动为LFQ tokens分配词汇表空间
- 提供编码/解码功能
- 支持二进制到token ID的转换

## 使用方法

### 基本用法

```python
from util.vocab_expansion import expand_vocab_for_lfq
from model.internvl.modeling_internvl_chat import InternVLChatModel

# 加载模型
model = InternVLChatModel.from_pretrained("path/to/model")

# 扩展词汇表
expanded_model = expand_vocab_for_lfq(
    model=model,
    lfq_config={
        'output_dim': 16  # LFQ输出维度
    },
    embedding_init_method="random",
    freeze_original=True  # 冻结原始embeddings
)
```

### 在joint_lfq.py中使用

```python
from runner.vq.joint_lfq import equip_internvl

# 加载配置
config = OmegaConf.load("config/vq/joint_lfq.yaml")

# 加载模型
model = InternVLChatModel.from_pretrained(config.model.internvl_path)

# 装备LFQ和扩展词汇表
equipped_model = equip_internvl(model, config)
```

## 配置选项

在`config/vq/joint_lfq.yaml`中：

```yaml
model:
  vocab_expansion:
    enabled: true
    embedding_init_method: "random"  # "random", "zeros", "mean"
    embedding_init_std: 0.02
    freeze_original: true  # 冻结原始embeddings，只训练新扩展部分
```

### 初始化方法

- `"random"`: 使用正态分布随机初始化
- `"zeros"`: 初始化为零
- `"mean"`: 使用已有embeddings的均值

## LFQ编码解码

### 编码LFQ codes为token IDs

```python
from util.vocab_expansion import encode_lfq_codes_to_tokens

# lfq_codes: (B, seq_len, output_dim) 二进制编码
token_ids = encode_lfq_codes_to_tokens(
    lfq_codes=lfq_codes,
    start_token_id=start_token_id,
    output_dim=16
)
```

### 解码token IDs为LFQ codes

```python
from util.vocab_expansion import decode_tokens_to_lfq_codes

# token_ids: (B, seq_len) token IDs
lfq_codes = decode_tokens_to_lfq_codes(
    token_ids=token_ids,
    start_token_id=start_token_id,
    output_dim=16
)
```

## 测试

运行测试脚本验证功能：

```bash
# 测试基本功能
python test_vocab_expansion.py

# 测试冻结功能
python test_freeze_embeddings.py
```

测试包括：
- 词汇表扩展功能
- LFQ编码解码
- Embedding保留
- 原始embeddings冻结
- 梯度冻结验证
- joint_lfq.py集成

## 技术细节

### 词汇表大小计算

对于LFQ，新的词汇表大小为：
```
new_vocab_size = original_vocab_size + 2^output_dim
```

### Token ID分配

- 原始tokens: 0 到 original_vocab_size-1
- LFQ tokens: original_vocab_size 到 new_vocab_size-1

### 二进制编码

LFQ使用二进制编码，每个维度对应一个bit：
```
token_id = sum(bit_i * 2^i) for i in range(output_dim)
```

## 注意事项

1. **内存使用**: 词汇表扩展会增加模型大小
2. **训练稳定性**: 新embeddings需要适当的初始化
3. **兼容性**: 确保tokenizer也支持新的词汇表大小
4. **保存/加载**: 扩展后的模型需要特殊处理
5. **冻结功能**: 冻结原始embeddings可以防止灾难性遗忘，但需要确保新embeddings有足够的训练数据

## 故障排除

### 常见问题

1. **词汇表大小不匹配**
   - 检查配置中的output_dim
   - 验证LFQ token范围计算

2. **Embedding形状错误**
   - 确保hidden_size一致
   - 检查padding_idx设置

3. **编码解码不一致**
   - 验证start_token_id
   - 检查output_dim参数

### 调试技巧

```python
# 检查词汇表大小
print(f"原始词汇表大小: {model.language_model.config.vocab_size}")

# 检查LFQ token范围
start_id, end_id = get_lfq_token_range(model, lfq_config)
print(f"LFQ token范围: {start_id} - {end_id}")

# 检查embedding形状
embeddings = model.language_model.get_input_embeddings()
print(f"Embedding形状: {embeddings.weight.shape}")

# 检查冻结状态
if hasattr(embeddings, 'trainable_mask'):
    trainable_count = embeddings.trainable_mask.sum().item()
    total_count = embeddings.trainable_mask.numel()
    print(f"可训练embedding比例: {trainable_count/total_count*100:.2f}%")

# 检查梯度冻结
if embeddings.weight.grad is not None:
    grad_norm = embeddings.weight.grad.norm().item()
    print(f"Embedding梯度范数: {grad_norm}")
```
