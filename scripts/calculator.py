


layers = 32
query_heads = 32
kv_heads = 8
head_dim = 128
context_length = 8192
# dtype = fp16

per_token_per_layer = kv_heads * head_dim * 2 * 2

print(context_length * layers * per_token_per_layer)
