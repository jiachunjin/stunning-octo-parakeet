"""
参数统计工具
用于统计模型中各个模块的可训练参数量
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def count_parameters(model: nn.Module, only_trainable: bool = True) -> Dict[str, int]:
    """
    统计模型中各个模块的参数数量
    
    Args:
        model: PyTorch模型
        only_trainable: 是否只统计可训练参数
    
    Returns:
        字典，键为模块名，值为参数数量
    """
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if only_trainable:
                param_counts[name] = trainable_params
            else:
                param_counts[name] = total_params
    
    return param_counts


def get_module_parameter_stats(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    获取详细的模块参数统计信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        字典，包含每个模块的详细参数统计
    """
    stats = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            stats[name] = {
                'total': total_params,
                'trainable': trainable_params,
                'frozen': frozen_params,
                'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
            }
    
    return stats


def print_parameter_summary(model: nn.Module, top_k: int = 20) -> None:
    """
    打印参数统计摘要
    
    Args:
        model: PyTorch模型
        top_k: 显示参数最多的前k个模块
    """
    stats = get_module_parameter_stats(model)
    
    # 按可训练参数数量排序
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['trainable'], reverse=True)
    
    print("=" * 80)
    print("模型参数统计摘要")
    print("=" * 80)
    
    total_trainable = sum(stat['trainable'] for stat in stats.values())
    total_frozen = sum(stat['frozen'] for stat in stats.values())
    total_params = total_trainable + total_frozen
    
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    print(f"冻结参数: {total_frozen:,} ({total_frozen/1e6:.2f}M)")
    print(f"可训练比例: {total_trainable/total_params*100:.2f}%")
    print()
    
    print(f"前{top_k}个参数最多的模块:")
    print("-" * 80)
    print(f"{'模块名':<50} {'可训练':<12} {'冻结':<12} {'总计':<12} {'比例':<8}")
    print("-" * 80)
    
    for i, (name, stat) in enumerate(sorted_stats[:top_k]):
        if stat['trainable'] > 0 or stat['frozen'] > 0:
            print(f"{name:<50} {stat['trainable']:>10,} {stat['frozen']:>10,} {stat['total']:>10,} {stat['trainable_ratio']:>6.1%}")
    
    print("-" * 80)


def get_embedding_stats(model: nn.Module) -> Dict[str, Dict[str, any]]:
    """
    获取embedding层的详细统计信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        embedding层的统计信息
    """
    embedding_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)) and ('embed' in name.lower() or 'lm_head' in name.lower()):
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            embedding_stats[name] = {
                'type': type(module).__name__,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': total_params - trainable_params,
                'shape': list(module.weight.shape) if hasattr(module, 'weight') else None,
                'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
            }
            
            # 如果有trainable_mask，显示冻结的token范围
            if hasattr(module, 'trainable_mask'):
                mask = module.trainable_mask
                frozen_count = (~mask).sum().item()
                trainable_count = mask.sum().item()
                embedding_stats[name]['frozen_tokens'] = frozen_count
                embedding_stats[name]['trainable_tokens'] = trainable_count
                embedding_stats[name]['total_tokens'] = mask.numel()
    
    return embedding_stats


def print_embedding_stats(model: nn.Module) -> None:
    """
    打印embedding层的统计信息
    
    Args:
        model: PyTorch模型
    """
    stats = get_embedding_stats(model)
    
    if not stats:
        print("未找到embedding层")
        return
    
    print("=" * 80)
    print("Embedding层参数统计")
    print("=" * 80)
    
    for name, stat in stats.items():
        print(f"模块: {name}")
        print(f"  类型: {stat['type']}")
        print(f"  形状: {stat['shape']}")
        print(f"  总参数: {stat['total_params']:,} ({stat['total_params']/1e6:.2f}M)")
        print(f"  可训练参数: {stat['trainable_params']:,} ({stat['trainable_params']/1e6:.2f}M)")
        print(f"  冻结参数: {stat['frozen_params']:,} ({stat['frozen_params']/1e6:.2f}M)")
        print(f"  可训练比例: {stat['trainable_ratio']:.2%}")
        
        if 'total_tokens' in stat:
            print(f"  总token数: {stat['total_tokens']:,}")
            print(f"  可训练token: {stat['trainable_tokens']:,}")
            print(f"  冻结token: {stat['frozen_tokens']:,}")
            print(f"  可训练token比例: {stat['trainable_tokens']/stat['total_tokens']:.2%}")
        
        print()


def get_lfq_stats(model: nn.Module) -> Dict[str, any]:
    """
    获取LFQ相关模块的统计信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        LFQ模块的统计信息
    """
    lfq_stats = {}
    
    # 查找LFQ相关模块
    for name, module in model.named_modules():
        if 'lfq' in name.lower() or hasattr(module, 'lfq'):
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            lfq_stats[name] = {
                'type': type(module).__name__,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': total_params - trainable_params,
                'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
            }
    
    # 检查LFQ token信息
    if hasattr(model, 'lfq_start_token_id') and hasattr(model, 'lfq_end_token_id'):
        lfq_stats['token_info'] = {
            'start_token_id': model.lfq_start_token_id,
            'end_token_id': model.lfq_end_token_id,
            'lfq_vocab_size': model.lfq_end_token_id - model.lfq_start_token_id
        }
    
    return lfq_stats


def print_lfq_stats(model: nn.Module) -> None:
    """
    打印LFQ相关模块的统计信息
    
    Args:
        model: PyTorch模型
    """
    stats = get_lfq_stats(model)
    
    if not stats:
        print("未找到LFQ相关模块")
        return
    
    print("=" * 80)
    print("LFQ模块参数统计")
    print("=" * 80)
    
    for name, stat in stats.items():
        if name == 'token_info':
            print(f"LFQ Token信息:")
            print(f"  起始token ID: {stat['start_token_id']}")
            print(f"  结束token ID: {stat['end_token_id']}")
            print(f"  LFQ词汇表大小: {stat['lfq_vocab_size']}")
            print()
        else:
            print(f"模块: {name}")
            print(f"  类型: {stat['type']}")
            print(f"  总参数: {stat['total_params']:,} ({stat['total_params']/1e6:.2f}M)")
            print(f"  可训练参数: {stat['trainable_params']:,} ({stat['trainable_params']/1e6:.2f}M)")
            print(f"  冻结参数: {stat['frozen_params']:,} ({stat['frozen_params']/1e6:.2f}M)")
            print(f"  可训练比例: {stat['trainable_ratio']:.2%}")
            print()


def comprehensive_model_stats(model: nn.Module) -> None:
    """
    打印模型的全面统计信息
    
    Args:
        model: PyTorch模型
    """
    print_parameter_summary(model)
    print_embedding_stats(model)
    print_lfq_stats(model)
