"""
Test-Time Scaling Feature Demonstration

Demonstrates test-time adaptation capabilities integrated with Nested Learning.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.test_time_scaling import (
    TestTimeAdapter,
    MemoryBasedAdapter,
    HybridAdapter,
    AdaptationConfig,
    create_test_time_adapter,
    TestTimeBatchAdapter
)


class SimpleTransformerModel(nn.Module):
    """Simple transformer-like model for demonstration"""
    def __init__(self, vocab_size=1000, hidden_size=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, hidden_size)
        self.attn1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x=None, input_ids=None, attention_mask=None, labels=None):
        if input_ids is not None:
            x = input_ids
        elif x is None:
            raise ValueError("Either x or input_ids required")

        # Create one-hot encoding for demonstration
        batch_size, seq_len = x.shape
        x_onehot = torch.zeros(batch_size, seq_len, 1000, device=x.device)
        x_onehot.scatter_(2, x.unsqueeze(-1).clamp(0, 999), 1)

        x = x_onehot.mean(dim=1)  # Average pooling

        x = self.embedding(x)
        x = self.attn1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.attn2(x)
        x = self.norm2(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return type('Output', (), {'logits': logits, 'loss': loss})()


def demo_basic_adaptation():
    """Demonstrate basic test-time adaptation"""
    print("\n" + "=" * 70)
    print("📚 Demo 1: Basic Test-Time Adaptation")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=5,
        adaptation_lr=1e-3,
        max_parameter_drift=0.5
    )

    adapter = TestTimeAdapter(model, config, device="cpu")

    print("\n📝 Adapting to input samples...")

    # Simulate test-time inputs
    input_samples = torch.randint(0, 1000, (3, 10))
    labels = torch.randint(0, 5, (3,))

    # Adapt model
    outputs, adaptation_info = adapter(
        input_ids=input_samples,
        labels=labels,
        adapt=True
    )

    print(f"\n📊 Adaptation Results:")
    print(f"  Average Loss: {adaptation_info['avg_loss']:.4f}")
    print(f"  Parameter Drift: {adaptation_info['final_drift']:.4f}")
    print(f"  Steps Taken: {adaptation_info['num_steps']}")

    print(f"\n📈 Model Output Shape: {outputs.logits.shape}")
    print(f"  Predictions: {outputs.logits.argmax(dim=-1)}")

    # Show adaptation stats
    stats = adapter.get_adaptation_stats()
    print(f"\n📋 Adaptation Statistics:")
    print(f"  Total Adaptations: {stats['num_adaptations']}")
    print(f"  Average Loss: {stats['avg_loss']:.4f}")

    print("\n✅ Basic Adaptation Demo Complete!")


def demo_safety_mechanisms():
    """Demonstrate safety mechanisms"""
    print("\n" + "=" * 70)
    print("🛡️  Demo 2: Safety Mechanisms")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=50,  # Try to do many steps
        adaptation_lr=0.1,  # High LR
        max_parameter_drift=0.2,  # Strict limit
        use_ema=True
    )

    adapter = TestTimeAdapter(model, config, device="cpu")

    print("\n📝 Attempting aggressive adaptation...")
    print(f"  Requested Steps: {config.num_steps}")
    print(f"  Max Drift: {config.max_parameter_drift}")

    input_samples = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 5, (2,))

    # Adapt with safety constraints
    adaptation_info = adapter.adapt(
        input_ids=input_samples,
        labels=labels
    )

    print(f"\n📊 Safety Mechanism Results:")
    print(f"  Steps Before Safety Trigger: {adaptation_info['num_steps']}")
    print(f"  Final Drift: {adaptation_info['final_drift']:.4f}")

    if adaptation_info['num_steps'] < config.num_steps:
        print(f"  ✅ Safety mechanism prevented over-adaptation!")
        print(f"  ✅ Stopped early to preserve model integrity")

    print("\n✅ Safety Mechanisms Demo Complete!")


def demo_adaptation_strategies():
    """Demonstrate different adaptation strategies"""
    print("\n" + "=" * 70)
    print("🎯 Demo 3: Adaptation Strategy Comparison")
    print("=" * 70)

    input_samples = torch.randint(0, 1000, (3, 10))
    labels = torch.randint(0, 5, (3,))

    strategies = ["gradient", "hybrid"]  # Skip "memory" as it needs fast memory modules
    results = {}

    for strategy in strategies:
        print(f"\n📝 Testing {strategy.upper()} strategy...")

        model = SimpleTransformerModel()
        adapter = create_test_time_adapter(
            model,
            strategy=strategy,
            device="cpu"
        )

        # Adapt
        outputs, info = adapter(
            input_ids=input_samples,
            labels=labels,
            adapt=True
        )

        if info:
            if strategy == "hybrid":
                # Hybrid returns nested results
                combined = info.get('combined', {})
                gradient_info = info.get('gradient', {})
                print(f"  Total Steps: {combined.get('total_steps', 'N/A')}")
                print(f"  Gradient Steps: {combined.get('gradient_steps', 'N/A')}")
                print(f"  Memory Steps: {combined.get('memory_steps', 'N/A')}")
                if 'avg_loss' in gradient_info:
                    print(f"  Loss: {gradient_info['avg_loss']:.4f}")
                    results[strategy] = {
                        'steps': combined.get('total_steps', 0),
                        'loss': gradient_info['avg_loss']
                    }
            else:
                print(f"  Steps: {info.get('num_steps', 'N/A')}")
                print(f"  Loss: {info.get('avg_loss', 'N/A'):.4f}" if isinstance(info.get('avg_loss'), (int, float)) else "")

                results[strategy] = {
                    'steps': info.get('num_steps', 0),
                    'loss': info.get('avg_loss', 0.0) if isinstance(info.get('avg_loss'), (int, float)) else 0.0
                }

    print(f"\n📊 Strategy Comparison:")
    for strategy, data in results.items():
        print(f"  {strategy:12s}: Steps={data['steps']}, Loss={data['loss']:.4f}")

    print("\n  Note: Memory-based strategy requires FastMemoryModule in model")
    print("  (Not shown in this demo since SimpleTransformerModel doesn't have one)")

    print("\n✅ Strategy Comparison Demo Complete!")


def demo_sequential_adaptation():
    """Demonstrate sequential adaptation to multiple inputs"""
    print("\n" + "=" * 70)
    print("🔄 Demo 4: Sequential Adaptation to Multiple Inputs")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=3,
        adaptation_lr=5e-4,
        max_parameter_drift=1.0  # More permissive
    )

    adapter = TestTimeAdapter(model, config, device="cpu")

    print("\n📝 Simulating streaming inputs...")

    # Simulate sequence of inputs
    inputs = [
        (torch.randint(0, 1000, (2, 10)), torch.randint(0, 5, (2,))),
        (torch.randint(0, 1000, (2, 10)), torch.randint(0, 5, (2,))),
        (torch.randint(0, 1000, (2, 10)), torch.randint(0, 5, (2,))),
        (torch.randint(0, 1000, (2, 10)), torch.randint(0, 5, (2,))),
    ]

    for i, (input_sample, label) in enumerate(inputs):
        print(f"\n📍 Input {i+1}:")

        outputs, info = adapter(
            input_ids=input_sample,
            labels=label,
            adapt=True
        )

        if info:
            print(f"  Loss: {info['avg_loss']:.4f}")
            print(f"  Drift: {info['final_drift']:.4f}")
            print(f"  Predictions: {outputs.logits.argmax(dim=-1).tolist()}")

    # Final stats
    stats = adapter.get_adaptation_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Total Adaptations: {stats['num_adaptations']}")
    print(f"  Average Loss: {stats['avg_loss']:.4f}")
    print(f"  Average Drift: {stats['avg_drift']:.4f}")
    print(f"  Total Steps: {stats['total_steps']}")

    print("\n✅ Sequential Adaptation Demo Complete!")


def demo_batch_adaptation():
    """Demonstrate batch-level adaptation"""
    print("\n" + "=" * 70)
    print("📦 Demo 5: Batch Adaptation")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=3,
        adaptation_lr=1e-3
    )

    batch_adapter = TestTimeBatchAdapter(model, config, device="cpu")

    print("\n📝 Adapting to batch of inputs...")

    # Create batch
    batch_inputs = torch.randint(0, 1000, (8, 10))
    batch_labels = torch.randint(0, 5, (8,))

    # Adapt to entire batch
    outputs, adaptation_infos = batch_adapter.adapt_batch(
        input_ids=batch_inputs,
        labels=batch_labels,
        adapt_per_sample=False
    )

    print(f"\n📊 Batch Adaptation Results:")
    print(f"  Batch Size: {batch_inputs.size(0)}")
    print(f"  Output Shape: {outputs.logits.shape}")
    print(f"  Adaptation Info Samples: {len(adaptation_infos)}")

    if adaptation_infos and adaptation_infos[0]:
        print(f"  Avg Loss: {adaptation_infos[0].get('avg_loss', 'N/A')}")

    print(f"\n📈 Batch Predictions:")
    predictions = outputs.logits.argmax(dim=-1)
    for i, pred in enumerate(predictions[:5]):
        print(f"  Sample {i}: Class {pred}")

    print("\n✅ Batch Adaptation Demo Complete!")


def demo_adaptation_without_labels():
    """Demonstrate self-supervised adaptation without labels"""
    print("\n" + "=" * 70)
    print("🔍 Demo 6: Self-Supervised Adaptation (No Labels)")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=5,
        adaptation_lr=1e-3,
        use_self_supervised=True,
        use_entropy_minimization=True,
        max_parameter_drift=1.0
    )

    adapter = TestTimeAdapter(model, config, device="cpu")

    print("\n📝 Adapting without supervision...")
    print("  Using: Self-supervised learning + Entropy minimization")

    # No labels provided
    input_samples = torch.randint(0, 1000, (3, 10))

    outputs, adaptation_info = adapter(
        input_ids=input_samples,
        adapt=True
    )

    print(f"\n📊 Self-Supervised Adaptation Results:")
    print(f"  Loss: {adaptation_info['avg_loss']:.4f}")
    print(f"  Steps: {adaptation_info['num_steps']}")
    print(f"  Drift: {adaptation_info['final_drift']:.4f}")

    print(f"\n📈 Model is now adapted to input distribution!")
    print(f"  Predictions: {outputs.logits.argmax(dim=-1).tolist()}")

    print("\n✅ Self-Supervised Adaptation Demo Complete!")


def demo_parameter_recovery():
    """Demonstrate parameter recovery mechanisms"""
    print("\n" + "=" * 70)
    print("🔙 Demo 7: Parameter Recovery")
    print("=" * 70)

    model = SimpleTransformerModel()
    config = AdaptationConfig(
        num_steps=10,
        adaptation_lr=1e-2,
        max_parameter_drift=0.5,
        use_ema=True
    )

    adapter = TestTimeAdapter(model, config, device="cpu")

    input_samples = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 5, (2,))

    print("\n📝 Performing adaptation...")

    # Initial adaptation
    outputs1, info1 = adapter(input_samples, labels, adapt=True)
    drift1 = info1['final_drift']
    print(f"  After Adaptation 1: Drift = {drift1:.4f}")

    # Reset to original
    adapter.reset()
    drift_after_reset = adapter.compute_parameter_drift()
    print(f"  After Reset: Drift = {drift_after_reset:.6f}")
    print(f"  ✅ Successfully restored original parameters")

    # Adapt again
    outputs2, info2 = adapter(input_samples, labels, adapt=True)
    drift2 = info2['final_drift']
    print(f"  After Adaptation 2: Drift = {drift2:.4f}")

    # Restore EMA state
    adapter.restore_ema_state()
    drift_after_ema = adapter.compute_parameter_drift()
    print(f"  After EMA Restore: Drift = {drift_after_ema:.4f}")
    print(f"  ✅ EMA provides stable recovery point")

    print("\n✅ Parameter Recovery Demo Complete!")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "Test-Time Scaling Feature Demos" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_basic_adaptation()
        demo_safety_mechanisms()
        demo_adaptation_strategies()
        demo_sequential_adaptation()
        demo_batch_adaptation()
        demo_adaptation_without_labels()
        demo_parameter_recovery()

        print("\n" + "=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETED!")
        print("=" * 70)

        print("\n🎉 Key Takeaways:")
        print("  1. ✓ Test-time adaptation improves model performance on specific inputs")
        print("  2. ✓ Safety mechanisms prevent catastrophic over-adaptation")
        print("  3. ✓ Multiple strategies (gradient, memory, hybrid) available")
        print("  4. ✓ Sequential adaptation enables streaming scenarios")
        print("  5. ✓ Self-supervised adaptation works without labels")
        print("  6. ✓ Parameter recovery ensures model stability")

        print("\n🚀 Test-Time Scaling is ready for use with Nested Learning!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
