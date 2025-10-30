import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

kv_cache = None

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]

        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)


def layer_norm(x, g_b, eps: float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input:
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    mean = x.mean(dim=-1, keepdim=True)
    std = torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + eps)
    x_norm = (x - mean) / std;
    return g * x_norm + b;


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer
        Input:
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    return x @ w + b


def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input:
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    x = linear(x, w_b1)
    x = gelu(x)
    x = linear(x, w_b2)

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input:
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # claculate scores
    d_k = k.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    # apply mask
    if mask is not None:
        scores = scores + mask
    attentions = softmax(scores)

    return attentions @ v


def mha(x, attn, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention

        Input:
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    n_seq, n_embed = x.shape
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    q, k, v = x.chunk(3, dim=-1)
    head_dim = n_embed // n_head
    q = q.view(n_seq, n_head, head_dim).transpose(0, 1)  ## [n_head, n_seq, head_dim]
    k = k.view(n_seq, n_head, head_dim).transpose(0, 1)
    v = v.view(n_seq, n_head, head_dim).transpose(0, 1)

    #cat cache
    if kv_cache is not None and kv_cache['k'] is not None:
        k = torch.cat([kv_cache['k'], k], dim=1)
        v = torch.cat([kv_cache['v'], v], dim=1)
    new_cache = {'k': k, 'v': v}
    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    total_len = k.size(1)
    causal_mask = torch.triu(torch.full((total_len, total_len), float('-inf')), diagonal=1)
    mask = causal_mask[-n_seq:, :]
    # Perform attention over each head
    out_heads = []  # n_head * [n_seq, n_embd/n_head]
    for h in range(n_head):
        out_h = attention(q[h], k[h], v[h], mask)
        out_heads.append(out_h)
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    out = torch.cat(out_heads, dim=-1)

    # Out projection
    out = linear(out, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return out, new_cache


def transformer_block(x, block, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    attn_out, new_cache = mha(layer_norm(x, ln_1), attn, n_head=n_head, kv_cache=kv_cache)
    x = x + attn_out # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, new_cache


def gpt2(inputs, params, n_head, kv_cache=None):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    is_first = kv_cache is None or kv_cache[0]['k'] is None
    if is_first:
        start = 0
        new_tokens = inputs
    else:
        start = kv_cache[0]['k'].size(1)
        new_tokens = inputs[-1:]

    pos = range(start, start + len(new_tokens))
    # token + positional embeddings
    x = wte[new_tokens] + wpe[pos]  # [n_seq] -> [n_seq, n_embd]

    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for i, block in enumerate(blocks):
        if kv_cache is None:
            current_cache = None
        else:
            current_cache = kv_cache[i]
        x, new_cache = transformer_block(x, block, n_head=n_head, kv_cache=current_cache)  # [n_seq, n_embd] -> [n_seq, n_embd]
        if kv_cache is not None:
            kv_cache[i] = new_cache
    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    global kv_cache
    kv_cache = [{'k': None, 'v': None} for _ in range(len(params['blocks']))]

    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head, kv_cache=kv_cache)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate:]  # only return generated ids


def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target,
                                n_tokens_to_generate, K):
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding

        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.

    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "1558M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)