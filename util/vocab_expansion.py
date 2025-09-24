"""
词汇表扩展工具
用于扩增模型的词汇表大小，同时保留已有的embeddings
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def expand_vocab_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    embedding_init_method: str = "random",
    embedding_init_std: float = 0.02,
    preserve_existing: bool = True,
    freeze_original: bool = True
) -> nn.Module:
    """
    扩展模型的词汇表大小，保留已有的embeddings
    
    Args:
        model: 要扩展的模型
        new_vocab_size: 新的词汇表大小
        embedding_init_method: 新embedding的初始化方法 ("random", "zeros", "mean")
        embedding_init_std: 随机初始化的标准差
        preserve_existing: 是否保留已有的embeddings
        freeze_original: 是否冻结原始embeddings，只训练新扩展的部分
    
    Returns:
        扩展后的模型
    """
    # 获取语言模型
    if hasattr(model, 'language_model'):
        language_model = model.language_model
    else:
        language_model = model
    
    # 获取当前词汇表大小
    old_vocab_size = language_model.config.vocab_size
    hidden_size = language_model.config.hidden_size
    
    if new_vocab_size <= old_vocab_size:
        logger.warning(f"新词汇表大小 ({new_vocab_size}) 不大于当前大小 ({old_vocab_size})")
        return model
    
    logger.info(f"扩展词汇表: {old_vocab_size} -> {new_vocab_size}")
    
    # 扩展input embeddings
    old_embed_tokens = language_model.get_input_embeddings()
    new_embed_tokens = nn.Embedding(new_vocab_size, hidden_size, old_embed_tokens.padding_idx)
    
    if preserve_existing:
        # 保留已有的embeddings
        with torch.no_grad():
            new_embed_tokens.weight[:old_vocab_size] = old_embed_tokens.weight
            # 初始化新的embeddings
            _init_new_embeddings(
                new_embed_tokens.weight[old_vocab_size:],
                method=embedding_init_method,
                std=embedding_init_std
            )
    
    # 设置新的input embeddings
    language_model.set_input_embeddings(new_embed_tokens)
    
    # 扩展output embeddings (lm_head)
    old_lm_head = language_model.get_output_embeddings()
    if old_lm_head is not None:
        new_lm_head = nn.Linear(hidden_size, new_vocab_size, bias=old_lm_head.bias is not None)
        
        if preserve_existing:
            with torch.no_grad():
                new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
                if old_lm_head.bias is not None:
                    new_lm_head.bias[:old_vocab_size] = old_lm_head.bias
                # 初始化新的权重
                _init_new_embeddings(
                    new_lm_head.weight[old_vocab_size:],
                    method=embedding_init_method,
                    std=embedding_init_std
                )
                if old_lm_head.bias is not None:
                    new_lm_head.bias[old_vocab_size:].zero_()
        
        # 设置新的output embeddings
        language_model.set_output_embeddings(new_lm_head)
    
    # 冻结原始embeddings，只训练新扩展的部分
    if freeze_original:
        _freeze_original_embeddings(language_model, old_vocab_size)
        logger.info(f"已冻结原始embeddings (0-{old_vocab_size-1})，只训练新扩展部分 ({old_vocab_size}-{new_vocab_size-1})")
    
    # 更新配置
    language_model.config.vocab_size = new_vocab_size
    
    logger.info(f"词汇表扩展完成，新大小: {new_vocab_size}")
    return model


def _init_new_embeddings(
    new_embeddings: torch.Tensor,
    method: str = "random",
    std: float = 0.02
) -> None:
    """
    初始化新的embeddings
    
    Args:
        new_embeddings: 新的embedding张量
        method: 初始化方法
        std: 随机初始化的标准差
    """
    with torch.no_grad():
        if method == "random":
            new_embeddings.normal_(mean=0.0, std=std)
        elif method == "zeros":
            new_embeddings.zero_()
        elif method == "mean":
            # 使用已有embeddings的均值
            new_embeddings.fill_(0.0)  # 这里可以改为已有embeddings的均值
        else:
            raise ValueError(f"不支持的初始化方法: {method}")


def _freeze_original_embeddings(language_model: nn.Module, old_vocab_size: int) -> None:
    """
    冻结原始embeddings，只让新扩展的部分可训练
    
    Args:
        language_model: 语言模型
        old_vocab_size: 原始词汇表大小
    """
    # 冻结input embeddings的原始部分
    input_embeddings = language_model.get_input_embeddings()
    if input_embeddings is not None:
        # 创建可训练mask
        input_embeddings.trainable_mask = torch.ones_like(input_embeddings.weight, dtype=torch.bool)
        input_embeddings.trainable_mask[:old_vocab_size] = False
        
        # 注册hook来冻结原始部分
        def freeze_input_hook(module, grad_input, grad_output):
            if hasattr(module, 'trainable_mask') and grad_input[0] is not None:
                # 将不可训练部分的梯度置零
                grad_input[0][~module.trainable_mask] = 0
            return grad_input
        
        input_embeddings.register_backward_hook(freeze_input_hook)
        logger.info(f"已设置input embeddings冻结mask: {old_vocab_size}个原始token被冻结")
    
    # 冻结output embeddings (lm_head)的原始部分
    output_embeddings = language_model.get_output_embeddings()
    if output_embeddings is not None:
        # 创建可训练mask
        output_embeddings.trainable_mask = torch.ones_like(output_embeddings.weight, dtype=torch.bool)
        output_embeddings.trainable_mask[:old_vocab_size] = False
        
        def freeze_output_hook(module, grad_input, grad_output):
            if hasattr(module, 'trainable_mask') and grad_input[0] is not None:
                # 将不可训练部分的梯度置零
                grad_input[0][~module.trainable_mask] = 0
            if len(grad_input) > 1 and grad_input[1] is not None and hasattr(module, 'trainable_mask'):  # bias
                # 对于bias，需要创建对应的mask
                bias_mask = module.trainable_mask[:, 0]  # 取第一列作为bias mask
                grad_input[1][~bias_mask] = 0
            return grad_input
        
        output_embeddings.register_backward_hook(freeze_output_hook)
        logger.info(f"已设置output embeddings冻结mask: {old_vocab_size}个原始token被冻结")


def expand_vocab_for_lfq(
    model: nn.Module,
    lfq_config: dict,
    embedding_init_method: str = "random",
    freeze_original: bool = True
) -> nn.Module:
    """
    为LFQ模型扩展词汇表
    
    Args:
        model: InternVL模型
        lfq_config: LFQ配置，包含output_dim等信息
        embedding_init_method: 新embedding的初始化方法
        freeze_original: 是否冻结原始embeddings，只训练新扩展的部分
    
    Returns:
        扩展后的模型
    """
    # 计算新的词汇表大小
    # 假设每个LFQ token需要2^output_dim个词汇表条目
    lfq_vocab_size = 2 ** lfq_config.get('output_dim', 16)
    
    # 获取当前词汇表大小
    current_vocab_size = model.language_model.config.vocab_size
    new_vocab_size = current_vocab_size + lfq_vocab_size
    
    logger.info(f"为LFQ扩展词汇表: {current_vocab_size} -> {new_vocab_size}")
    logger.info(f"LFQ词汇表大小: {lfq_vocab_size}")
    
    # 扩展词汇表
    model = expand_vocab_embeddings(
        model=model,
        new_vocab_size=new_vocab_size,
        embedding_init_method=embedding_init_method,
        preserve_existing=True,
        freeze_original=freeze_original
    )
    
    return model


def get_lfq_token_range(
    model: nn.Module,
    lfq_config: dict
) -> tuple:
    """
    获取LFQ token的词汇表范围
    
    Args:
        model: 扩展后的模型
        lfq_config: LFQ配置
    
    Returns:
        (start_token_id, end_token_id) LFQ token的ID范围
    """
    current_vocab_size = model.language_model.config.vocab_size
    lfq_vocab_size = 2 ** lfq_config.get('output_dim', 16)
    
    start_token_id = current_vocab_size - lfq_vocab_size
    end_token_id = current_vocab_size
    
    return start_token_id, end_token_id


def encode_lfq_codes_to_tokens(
    lfq_codes: torch.Tensor,
    start_token_id: int,
    output_dim: int
) -> torch.Tensor:
    """
    将LFQ二进制编码转换为token IDs
    
    Args:
        lfq_codes: LFQ二进制编码 (B, seq_len, output_dim)
        start_token_id: LFQ token的起始ID
        output_dim: LFQ输出维度
    
    Returns:
        token_ids: 对应的token IDs (B, seq_len)
    """
    # 将二进制编码转换为整数
    # lfq_codes: (B, seq_len, output_dim) -> (B, seq_len)
    token_ids = torch.zeros(lfq_codes.shape[:2], dtype=torch.long, device=lfq_codes.device)
    
    for i in range(output_dim):
        token_ids += (lfq_codes[..., i] > 0).long() * (2 ** i)
    
    # 加上起始token ID
    token_ids += start_token_id
    
    return token_ids


def decode_tokens_to_lfq_codes(
    token_ids: torch.Tensor,
    start_token_id: int,
    output_dim: int
) -> torch.Tensor:
    """
    将token IDs解码为LFQ二进制编码
    
    Args:
        token_ids: token IDs (B, seq_len)
        start_token_id: LFQ token的起始ID
        output_dim: LFQ输出维度
    
    Returns:
        lfq_codes: LFQ二进制编码 (B, seq_len, output_dim)
    """
    # 减去起始token ID
    relative_ids = token_ids - start_token_id
    
    # 转换为二进制编码
    lfq_codes = torch.zeros(
        token_ids.shape[0], token_ids.shape[1], output_dim,
        dtype=torch.float, device=token_ids.device
    )
    
    for i in range(output_dim):
        lfq_codes[..., i] = ((relative_ids >> i) & 1).float()
    
    return lfq_codes
