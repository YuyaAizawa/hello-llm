import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from twilog_data_loader import TwilogDataLoader


torch.manual_seed(114514)
batch_size = 128
block_size = 142
max_iters = 1001
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 512
n_head = 16
n_layer = 8

print(f"device: {device}")

class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    assert head_size % 2 == 0
    theta = 1. / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    pos = torch.arange(block_size)
    freqs = pos.unsqueeze(1) * theta.unsqueeze(0)
    freqs = freqs.unsqueeze(-1).expand(block_size, head_size // 2, 2).reshape(block_size, -1)
    self.register_buffer('freqs', freqs)

  def rot_half(self, x):                      # [1, 2, 3, 4]
    d = x.shape[-1] // 2
    x = x.view(*x.shape[:-1], d, 2)     # [[1, 2], [3, 4]]
    x1, x2 = x.unbind(dim=-1)           # [1, 3], [2, 4]
    x = torch.stack((-x2, x1), dim=-1)  # [[-2, 1], [-4, 3]]
    x = x.view(*x.shape[:-2], d * 2)    # [-2, 1, -4, 3]
    return x

  def rot(self, x):
    freqs = self.freqs[:x.shape[-2], :]
    return x * freqs.cos() + self.rot_half(x) * freqs.sin()

  def forward(self, x, attn_mask=None):
    B, T, C = x.shape
    k = self.key(x)  # [B*T*H]
    q = self.query(x)  # [B*T*H]
    k = self.rot(k)
    q = self.rot(k)
    wei = q @ k.transpose(-2, -1) * C**-0.5  # [B*T*T]
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    if attn_mask is not None:
      wei += attn_mask
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)  # [B*T*H]
    out = wei @ v  # [B*T*H]
    return out

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)

  def forward(self, x, attn_mask=None):
    out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedFoward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
    )

  def forward(self, x):
    x = self.net(x)
    return x

class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.norm1 = nn.LayerNorm(n_embd)
    self.sa = MultiHeadAttention(n_head, head_size)
    self.norm2 = nn.LayerNorm(n_embd)
    self.ffwd = FeedFoward(n_embd)

  def forward(self, x, attn_mask=None):
    identity = x
    x = self.norm1(x)
    x = self.sa(x, attn_mask)
    x += identity
    x = self.norm2(x)
    identity = x
    x = self.ffwd(x)
    x += identity
    return x

class LanguageModel(nn.Module):

  def __init__(self, vocab_size, device='cpu'):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.blocks = nn.ModuleList()
    for _ in range(n_layer):
      self.blocks.append(Block(n_embd, n_head=n_head))
    self.norm = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None, attn_mask=None):
    B, T = idx.shape

    if attn_mask is not None:
      attn_mask = attn_mask[:, None, :]
      attn_mask = attn_mask.to(dtype=torch.float)
      attn_mask = torch.where(attn_mask == 0, float('-inf'), 0.0)

    x = self.token_embedding_table(idx)  # (B, T, C)
    for block in self.blocks:
      x = block(x, attn_mask)  # (B, T, C)
    x = self.norm(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets is None:
      loss = None
    else :
      B, T, V = logits.shape
      logits = logits.view(B*T, V)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets, ignore_index=2)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)  # TODO beam search
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y, mask = tweets.get_batch(batch_size, split)
      logits, loss = model(X, Y, mask)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# logger
import logging
from datetime import datetime
logger = logging.getLogger('twilog')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
# for console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# for file
file_handler = logging.FileHandler('twilog_{}.log'.format(datetime.utcnow().strftime("%Y%m%d%H%M")))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f'batch_size = {batch_size}, block_size = {block_size}, learning_rate = {learning_rate}, max_iters = {max_iters}')

tweets = TwilogDataLoader('data/TypedTypelessTy-240615.csv', block_size, device)
model = LanguageModel(tweets.vocab_size)
model = model.to(device)

logger.info(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'total_params = {total_params}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

  if iter % eval_interval == 0:
    losses = estimate_loss()
    logger.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb, attn_mask = tweets.get_batch(batch_size, 'train')

  logits, loss = model(xb, yb, attn_mask)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((16, 1), dtype=torch.long, device=device)
results = model.generate(context, max_new_tokens=140)
for result in results:
  logger.info(tweets.decode(result.tolist()).split('<eot>')[0])
