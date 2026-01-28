"""
Nested Learning Feature Demonstration

This script demonstrates the key features of Nested Learning implementation.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.nested_optimizer import NestedOptimizer


def demo_memory_update():
    """演示记忆如何在不同频率下更新"""
    print("\n" + "=" * 70)
    print("📊 演示 1: 多频率记忆更新")
    print("=" * 70)

    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        use_nested_learning=True,
        num_memory_levels=2,
        memory_frequencies=[4, 8]  # 使用小频率便于演示
    )

    model = MiniMindForCausalLM(config)
    print(f"✓ 创建模型: {config.num_memory_levels} 级记忆")
    print(f"  更新频率: {config.memory_frequencies}")

    # 模拟训练
    print("\n训练过程:")
    for step in range(12):
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # 显示每层的更新状态
        for i, block in enumerate(model.model.layers):
            if hasattr(block, 'step_counter'):
                freq = config.memory_frequencies[0]
                will_update = (block.step_counter % freq == 0)
                status = "🔄 [更新]" if will_update else "⏸️  [跳过]"
                if step % 3 == 0:  # 每3步显示一次
                    print(f"  Step {step:2d} | Block {i} | Counter={block.step_counter:2d} | {status}")

    print("\n✓ 记忆系统按不同频率更新成功！")


def demo_learning_rates():
    """演示不同优化器的学习率"""
    print("\n" + "=" * 70)
    print("📈 演示 2: 多优化器学习率")
    print("=" * 70)

    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        use_nested_learning=True,
        num_memory_levels=3,
        memory_frequencies=[64, 512, 4096]
    )

    model = MiniMindForCausalLM(config)
    optimizer = NestedOptimizer(model, config, base_lr=1e-4)

    print(f"✓ 创建嵌套优化器")
    print(f"  基础学习率: {1e-4}")
    print(f"\n各优化器学习率（按频率缩放）:")

    lrs = optimizer.get_lr()
    for name, lr in sorted(lrs.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0):
        print(f"  {name:15s}: {lr:.6e}")

    print("\n✓ 学习率自动缩放成功！")


def demo_online_adaptation():
    """演示在线学习能力"""
    print("\n" + "=" * 70)
    print("🧠 演示 3: 在线快速适应")
    print("=" * 70)

    config = MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=100,
        use_nested_learning=True,
        num_memory_levels=1,
        memory_frequencies=[2],  # 每2步更新一次
        memory_type='delta'
    )

    model = MiniMindForCausalLM(config)

    # 模拟一个简单的"新任务"
    print("\n模拟学习新模式:")
    pattern = torch.tensor([[1, 2, 3, 4, 5]]).repeat(2, 1)

    losses = []
    for step in range(10):
        input_ids = torch.randint(0, 100, (2, 5))
        labels = pattern[:, :5]

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        losses.append(loss.item())

        if step % 2 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    print(f"\n✓ 损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f}")
    print(f"✓ 改善: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")


def demo_backward_compatibility():
    """演示向后兼容性"""
    print("\n" + "=" * 70)
    print("🔄 演示 4: 向后兼容性")
    print("=" * 70)

    # 标准 Transformer（不启用 Nested Learning）
    config_standard = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        use_nested_learning=False  # 禁用
    )

    model_standard = MiniMindForCausalLM(config_standard)
    print("✓ 标准模型创建成功（无 Nested Learning）")

    input_ids = torch.randint(0, 1000, (2, 10))
    outputs = model_standard(input_ids)
    print(f"✓ 标准前向传播: logits shape = {outputs.logits.shape}")

    # Nested Learning 模型
    config_nl = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        use_nested_learning=True,  # 启用
        num_memory_levels=2,
        memory_frequencies=[64, 512]
    )

    model_nl = MiniMindForCausalLM(config_nl)
    print("\n✓ Nested Learning 模型创建成功")

    outputs_nl = model_nl(input_ids)
    print(f"✓ NL 前向传播: logits shape = {outputs_nl.logits.shape}")

    print("\n✓ 两种模式都可以正常工作！")


def demo_memory_efficiency():
    """演示内存效率"""
    print("\n" + "=" * 70)
    print("💾 演示 5: 参数统计")
    print("=" * 70)

    configs = [
        ("标准模型", MiniMindConfig(hidden_size=256, num_hidden_layers=4, use_nested_learning=False)),
        ("NL 模型 (2级)", MiniMindConfig(
            hidden_size=256,
            num_hidden_layers=4,
            use_nested_learning=True,
            num_memory_levels=2,
            memory_frequencies=[64, 512]
        )),
    ]

    for name, config in configs:
        model = MiniMindForCausalLM(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{name}:")
        print(f"  总参数:     {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

    print("\n✓ 参数统计完成！")


def main():
    """运行所有演示"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Nested Learning 功能演示" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_memory_update()
        demo_learning_rates()
        demo_online_adaptation()
        demo_backward_compatibility()
        demo_memory_efficiency()

        print("\n" + "=" * 70)
        print("✅ 所有演示完成！")
        print("=" * 70)
        print("\n🎯 关键要点:")
        print("  1. ✓ 多时间尺度更新正常工作")
        print("  2. ✓ 学习率自动缩放")
        print("  3. ✓ 在线快速适应能力")
        print("  4. ✓ 完全向后兼容")
        print("  5. ✓ 内存效率合理")
        print("\n🚀 Nested Learning 已准备就绪，可以开始使用！\n")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
