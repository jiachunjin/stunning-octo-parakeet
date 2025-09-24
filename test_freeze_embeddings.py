#!/usr/bin/env python3
"""
测试embedding冻结功能
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from model.internvl.modeling_internvl_chat import InternVLChatModel
from util.vocab_expansion import expand_vocab_for_lfq


def test_embedding_freeze():
    """测试embedding冻结功能"""
    print("=" * 50)
    print("测试embedding冻结功能")
    print("=" * 50)
    
    # 加载配置
    config_path = "config/vq/joint_lfq.yaml"
    config = OmegaConf.load(config_path)
    
    print(f"配置路径: {config_path}")
    print(f"LFQ输出维度: {config.model.down_proj.output_dim}")
    
    # 加载模型
    print("\n加载InternVL模型...")
    try:
        model = InternVLChatModel.from_pretrained(config.model.internvl_path)
        original_vocab_size = model.language_model.config.vocab_size
        print(f"原始词汇表大小: {original_vocab_size}")
        
        # 扩展词汇表并冻结原始部分
        print("\n执行词汇表扩展并冻结原始embeddings...")
        expanded_model = expand_vocab_for_lfq(
            model=model,
            lfq_config=config.model.down_proj,
            embedding_init_method="random",
            freeze_original=True
        )
        
        new_vocab_size = expanded_model.language_model.config.vocab_size
        print(f"扩展后词汇表大小: {new_vocab_size}")
        
        # 测试梯度冻结
        print("\n测试梯度冻结...")
        test_gradient_freeze(expanded_model, original_vocab_size)
        
        print("\n✓ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_gradient_freeze(model, original_vocab_size):
    """测试梯度冻结功能"""
    print("测试梯度冻结...")
    
    # 创建虚拟输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model.language_model.config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # 前向传播
    outputs = model.language_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    # 反向传播
    loss.backward()
    
    # 检查input embeddings的梯度
    input_embeddings = model.language_model.get_input_embeddings()
    if input_embeddings.weight.grad is not None:
        grad = input_embeddings.weight.grad
        original_grad = grad[:original_vocab_size]
        new_grad = grad[original_vocab_size:]
        
        # 检查原始部分的梯度是否被冻结
        original_grad_norm = original_grad.norm().item()
        new_grad_norm = new_grad.norm().item()
        
        print(f"原始embeddings梯度范数: {original_grad_norm:.6f}")
        print(f"新embeddings梯度范数: {new_grad_norm:.6f}")
        
        if original_grad_norm < 1e-8:
            print("✓ 原始embeddings梯度已被正确冻结")
        else:
            print("❌ 原始embeddings梯度未被冻结")
        
        if new_grad_norm > 1e-8:
            print("✓ 新embeddings梯度正常")
        else:
            print("❌ 新embeddings梯度异常")
    
    # 检查output embeddings的梯度
    output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings.weight.grad is not None:
        grad = output_embeddings.weight.grad
        original_grad = grad[:original_vocab_size]
        new_grad = grad[original_vocab_size:]
        
        # 检查原始部分的梯度是否被冻结
        original_grad_norm = original_grad.norm().item()
        new_grad_norm = new_grad.norm().item()
        
        print(f"原始output embeddings梯度范数: {original_grad_norm:.6f}")
        print(f"新output embeddings梯度范数: {new_grad_norm:.6f}")
        
        if original_grad_norm < 1e-8:
            print("✓ 原始output embeddings梯度已被正确冻结")
        else:
            print("❌ 原始output embeddings梯度未被冻结")
        
        if new_grad_norm > 1e-8:
            print("✓ 新output embeddings梯度正常")
        else:
            print("❌ 新output embeddings梯度异常")


def test_trainable_parameters():
    """测试可训练参数统计"""
    print("\n" + "=" * 50)
    print("测试可训练参数统计")
    print("=" * 50)
    
    try:
        from runner.vq.joint_lfq import equip_internvl
        from omegaconf import OmegaConf
        
        # 加载配置
        config = OmegaConf.load("config/vq/joint_lfq.yaml")
        
        # 加载模型
        model = InternVLChatModel.from_pretrained(config.model.internvl_path)
        original_vocab_size = model.language_model.config.vocab_size
        
        print(f"原始词汇表大小: {original_vocab_size}")
        
        # 装备模型
        equipped_model = equip_internvl(model, config.model)
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in equipped_model.parameters())
        trainable_params = sum(p.numel() for p in equipped_model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
        
        # 检查LFQ相关参数
        if hasattr(equipped_model, 'lfq'):
            lfq_params = sum(p.numel() for p in equipped_model.lfq.parameters() if p.requires_grad)
            print(f"LFQ可训练参数量: {lfq_params:,}")
        
        # 检查embedding参数
        input_embeddings = equipped_model.language_model.get_input_embeddings()
        output_embeddings = equipped_model.language_model.get_output_embeddings()
        
        if hasattr(input_embeddings, 'trainable_mask'):
            trainable_embeddings = input_embeddings.trainable_mask.sum().item()
            total_embeddings = input_embeddings.trainable_mask.numel()
            print(f"Input embeddings可训练比例: {trainable_embeddings/total_embeddings*100:.2f}%")
        
        if hasattr(output_embeddings, 'trainable_mask'):
            trainable_output = output_embeddings.trainable_mask.sum().item()
            total_output = output_embeddings.trainable_mask.numel()
            print(f"Output embeddings可训练比例: {trainable_output/total_output*100:.2f}%")
        
        print("✓ 参数统计测试通过")
        
    except Exception as e:
        print(f"❌ 参数统计测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行所有测试
    test_embedding_freeze()
    test_trainable_parameters()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
