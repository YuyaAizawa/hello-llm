import csv
import os.path as osp
import torch
import random


class TwilogDataLoader():

  def __init__(self, csv_path, block_size, device='cpu'):
    self.block_size = block_size
    self.device = device

    tweets = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
      csvreader = csv.reader(csvfile)
      for row in csvreader:
        tweet = row[3]

        # remove reply
        if tweet.startswith('@') and ' ' in tweet:
          tweet = tweet.split(' ', 1)[1]

        # remove link
        chunks = [ chunk for chunk in tweet.split(' ') if not chunk.startswith('http') ]
        tweet = ' '.join(chunks)
        chunks = [ chunk for chunk in tweet.split('\n') if not chunk.startswith('http') ]
        tweet = '\n'.join(chunks)

        if 0 < len(tweet) and len(tweet) <= 140:
          tweets.append(tweet)

    char_set = set()
    for tweet in tweets:
      char_set.update(tweet)
    chars = sorted(list(char_set))

    self.vocab_size = len(chars) + 3
    self.stoi = { ch:i+3 for i, ch in enumerate(chars) }
    self.itos = { i+3:ch for i, ch in enumerate(chars) }
    self.itos.update({0:'<bot>', 1:'<eot>', 2:'<pad>' })

    data = []
    long_tweet = 0
    for tweet in tweets:
      data.append(0)
      data.extend([self.stoi[c] for c in tweet])
      data.append(1)
      data.extend([2] * (block_size - (len(tweet) + 2) + 1))
    data = torch.tensor(data, device=device, dtype=torch.long)
    data = data.view(len(tweets), block_size + 1)

    n = int(0.9*len(data))
    self.train_data = data[:n]
    self.val_data = data[n:]


  def get_batch(self, batch_size, split=None):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data), (batch_size, ))
    tweets = torch.stack([data[i] for i in ix])

    x = tweets[:, :-1].contiguous()
    y = tweets[:, 1:].contiguous()
    attn_mask = torch.where(x == 2, 0, 1)

    return x.to(self.device), y.to(self.device), attn_mask.to(self.device)


  def decode(self, l):
    return ''.join([self.itos[i] for i in l])
