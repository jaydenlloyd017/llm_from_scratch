"""
Multi-head attention: shape walkthrough

Runs one multi-head attention pass step by step and logs the tensor
shape after each operation, to out/mha_shapes.txt.

NOTE: The input (torch.randn) and the layer weights are random and
untrained, so the actual numbers and attention weights are meaningless.
This demo shows the MECHANISM (how data is reshaped, split into heads,
scored and recombined), not learned behaviour. The model only learns
what to pay attention to during training in a soon to be made file.

Example config: B=1 sentence, T=5 tokens, d_model=12, n_head=3 (d_head=4)
"""

import os
import math
import torch
from multi_head import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), 'out', 'mha_shapes.txt')

def log(s):
    print(s)
    with open(OUT_TXT, 'a') as f:
        f.write(s + '\n')

if __name__ == "__main__":
    # Reset file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, 'w').close()

    B, T, d_model, n_head = 1, 5, 12, 3
    d_head = d_model // n_head
    x = torch.randn(B, T, d_model)              # Step 0: the input
    # One table: 5 rows (words) × 12 numbers
        # The    [101010101010]
        # cat    [101010101010]
        # sat    [101010101010]
        # it     [101010101010]
        # tired  [101010101010] 
    
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)

    log(f"Input x:           {tuple(x.shape)} = (B,T,d_model)")
    qkv = attn.qkv(x)  # (B,T,3*d_model)                                Step 1: make Q, K and V
    # Each row grows to 36 numbers: 12 Q, 12 K, 12 V
        # The    [QQQQQQQQQQQQ KKKKKKKKKKKK VVVVVVVVVVVV]
        # cat    [QQQQQQQQQQQQ KKKKKKKKKKKK VVVVVVVVVVVV]
        # sat    [QQQQQQQQQQQQ KKKKKKKKKKKK VVVVVVVVVVVV]
        # it     [QQQQQQQQQQQQ KKKKKKKKKKKK VVVVVVVVVVVV]
        # tired  [QQQQQQQQQQQQ KKKKKKKKKKKK VVVVVVVVVVVV]
    log(f"Linear qkv(x):     {tuple(qkv.shape)} = (B,T,3*d_model)")

    qkv = qkv.view(B, T, 3, n_head, d_head)
    # Each row's 36 numbers get labelled by head: 1, 2, 3               Step 2: sort into boxes (no numbers change)
    # The    Q[1111|2222|3333]  K[1111|2222|3333]  V[1111|2222|3333]
    # cat    Q[1111|2222|3333]  K[1111|2222|3333]  V[1111|2222|3333]
    # sat    Q[1111|2222|3333]  K[1111|2222|3333]  V[1111|2222|3333]
    # it     Q[1111|2222|3333]  K[1111|2222|3333]  V[1111|2222|3333]
    # tired  Q[1111|2222|3333]  K[1111|2222|3333]  V[1111|2222|3333]
    log(f"view to 5D:        {tuple(qkv.shape)} = (B,T,3,heads,d_head)")

    q, k, v = qkv.unbind(dim=2)
    # Three separate tables. Here's q (k and v look the same):          Step 3: cut out Q, K and V
    # q:
    # The    [1111|2222|3333]
    # cat    [1111|2222|3333]
    # sat    [1111|2222|3333]
    # it     [1111|2222|3333]
    # tired  [1111|2222|3333]
    log(f"q,k,v split:       q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    q = q.transpose(1, 2)
    # Pull each head's column out into its own table:           Step 4: group by head 
    # q, Head 1        q, Head 2        q, Head 3
    # The   [1111]     The   [2222]     The   [3333]
    # cat   [1111]     cat   [2222]     cat   [3333]
    # sat   [1111]     sat   [2222]     sat   [3333]
    # it    [1111]     it    [2222]     it    [3333]
    # tired [1111]     tired [2222]     tired [3333]
    # Now each head has all 5 words in one place. The same happens to k and v.
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    log(f"transpose heads:   q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} = (B,heads,T,d_head)")

    scale = 1.0 / math.sqrt(d_head)                         # Step 5: every word scores every word
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    # Each head makes a 5×5 grid: row = the word asking, column = the word being checked
        # Head 1:   The cat sat  it tired
        # The      [ ■   ■   ■   ■   ■ ]
        # cat      [ ■   ■   ■   ■   ■ ]
        # sat      [ ■   ■   ■   ■   ■ ]
        # it       [ ■   ■   ■   ■   ■ ]   ← how much "it" matches each word
        # tired    [ ■   ■   ■   ■   ■ ]
        # (same for Head 2 and Head 3)
    log(f"scores q@k^T:      {tuple(scores.shape)} = (B,heads,T,T)")


    weights = torch.softmax(scores, dim=-1)     
    # Same grid shape, but each row now adds up to 100%                             # Step 6: turn scores into percentages
    # Head 1:   The  cat  sat  it  tired
    # it       [ 2%  80%  5%  6%   7% ]    ← "it" would mostly looks at "cat" after training. 
    log(f"softmax(weights):  {tuple(weights.shape)} = (B,heads,T,T)")

    ctx = torch.matmul(weights, v)                                                  # Step 7: gather information
    # Each word becomes a mix of the other words' V, using those percentages. 
    # The shape goes back to tables of 5 words × 4:
    # ctx, Head 1      ctx, Head 2      ctx, Head 3
    # The   [1111]     The   [2222]     The   [3333]
    # cat   [1111]     cat   [2222]     cat   [3333]
    # ...              ...              ...
    # it    [1111]  ← would now contain "cat" info after training.
    log(f"context @v:        {tuple(ctx.shape)} = (B,heads,T,d_head)")

    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)              # Step 8: put the heads back side by side
    # This undoes step 4: glue the 3 head tables back into one
    # The    [1111|2222|3333]
    # cat    [1111|2222|3333]
    # sat    [1111|2222|3333]
    # it     [1111|2222|3333]
    # tired  [1111|2222|3333]
    log(f"merge heads:       {tuple(out.shape)} = (B,T,d_model)")

    out = attn.proj(out)                                                    # Step 9: final mix
    # The heads' results get blended together. We're back to 5 words × 12 numbers, the same shape as step 0, 
    # but each word now knows about the other words.
    # The    [■■■■■■■■■■■■]
    # cat    [■■■■■■■■■■■■]
    # sat    [■■■■■■■■■■■■]
    # it     [■■■■■■■■■■■■]   ← after training, "it" would mostly look at "cat"
    # tired  [■■■■■■■■■■■■]
    log(f"final proj:        {tuple(out.shape)} = (B,T,d_model)")

    log("\nLegend:")
    log("  B=batch, T=sequence length, d_model=embedding size, heads=n_head, d_head=d_model/heads")
    log("  qkv(x) is a single Linear producing [Q|K|V]; we reshape then split into q,k,v")