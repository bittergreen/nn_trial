from core.model4 import MiniGPT, n_embd, context_dim, n_layer, n_head, head_size
import torch

def count_parameters(model):
    """Count total trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def format_param_count(count):
    """Format parameter count in human readable form"""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)

def analyze_model_size():
    # Assume a reasonable vocab size for analysis
    vocab_size = 50000  # Common vocab size
    
    print(f"Model Configuration:")
    print(f"- n_embd: {n_embd}")
    print(f"- n_layer: {n_layer}")
    print(f"- n_head: {n_head}")
    print(f"- head_size: {head_size}")
    print(f"- context_dim: {context_dim}")
    print(f"- vocab_size: {vocab_size}")
    print()
    
    # Create model instance
    model = MiniGPT(vocab_size)
    
    # Count total parameters
    total_params = count_parameters(model)
    
    print(f"Total Parameters: {format_param_count(total_params)} ({total_params:,})")
    print()
    
    # Break down by component
    print("Parameter Breakdown:")
    
    # Embeddings
    token_emb_params = vocab_size * n_embd
    pos_emb_params = 256 * n_embd  # sequence_length = 256
    print(f"- Token Embedding: {format_param_count(token_emb_params)}")
    print(f"- Position Embedding: {format_param_count(pos_emb_params)}")
    
    # Transformer blocks (5 early blocks)
    # Each block: self-attention + FFN + layer norms
    # Self-attention: 4 linear layers (q,k,v,out) each n_embd x n_embd
    # FFN: 2 linear layers (n_embd x 4*n_embd, 4*n_embd x n_embd)
    # Layer norms: 2 x n_embd parameters each
    
    sa_params_per_block = 4 * (n_embd * n_embd + n_embd)  # +bias
    ffn_params_per_block = (n_embd * 4 * n_embd + 4 * n_embd) + (4 * n_embd * n_embd + n_embd)
    ln_params_per_block = 2 * 2 * n_embd  # 2 layer norms, each has weight+bias
    
    early_block_params = 5 * (sa_params_per_block + ffn_params_per_block + ln_params_per_block)
    print(f"- Early Blocks (5x): {format_param_count(early_block_params)}")
    
    # Hippocampus
    # ContextMerger: n_embd -> context_dim
    context_merger_params = n_embd * context_dim + context_dim
    # NeuralMemtable: 2 linear layers + memory parameters
    memtable_linear_params = 2 * (context_dim * context_dim + context_dim)
    memtable_memory_params = 64 * context_dim  # 64 memory slots
    # Output projection: context_dim -> n_embd
    hippo_out_params = context_dim * n_embd + n_embd
    
    hippo_total = context_merger_params + memtable_linear_params + memtable_memory_params + hippo_out_params
    print(f"- Hippocampus: {format_param_count(hippo_total)}")
    
    # Final block (with cross-attention)
    cross_attn_params = 4 * (n_embd * n_embd + n_embd)  # q,k,v,out for cross-attention
    final_block_params = sa_params_per_block + cross_attn_params + ffn_params_per_block + 3 * 2 * n_embd  # 3 layer norms
    print(f"- Final Block (with cross-attn): {format_param_count(final_block_params)}")
    
    # Output head
    lm_head_params = n_embd * vocab_size + vocab_size
    final_ln_params = 2 * n_embd
    print(f"- LM Head: {format_param_count(lm_head_params)}")
    print(f"- Final LayerNorm: {format_param_count(final_ln_params)}")
    
    print()
    print("Comparison to Standard LLMs:")
    print("- GPT-2 Small: 117M parameters")
    print("- GPT-2 Medium: 345M parameters")
    print("- GPT-2 Large: 762M parameters")
    print("- GPT-2 XL: 1.5B parameters")
    print("- LLaMA 7B: 7B parameters")
    print("- LLaMA 13B: 13B parameters")
    
    # Classification
    if total_params < 1e6:
        size_class = "Tiny/Experimental"
    elif total_params < 100e6:
        size_class = "Small"
    elif total_params < 1e9:
        size_class = "Medium"
    elif total_params < 10e9:
        size_class = "Large"
    else:
        size_class = "Very Large"
    
    print(f"\nYour model4 is: {size_class} ({format_param_count(total_params)})")

if __name__ == "__main__":
    analyze_model_size()