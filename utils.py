import torch
import json
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch(batch_size, data, max_length):
    ix = torch.randint(len(data)-max_length, (batch_size, ))
    x = torch.stack([data[i:i+max_length] for i in ix])
    y = torch.stack([data[i+1:i+max_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return (x, y)


@torch.no_grad()
def estimate_loss(model, data_iter, data_test_iter, max_length, eval_iters, tokenizer):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        k = 0
        while k < eval_iters:
            if split == 'train':
                batch = next(data_iter)
            else:
                batch = next(data_test_iter)
            batch = next(data_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            while (input_ids.shape[1] > 0 and k < eval_iters):
                if input_ids.shape[1] <= max_length:
                    x = batch['input_ids'][:, -max_length-1:-1]
                    x_mask = batch['attention_mask'][:, -max_length-1:-1]
                    y = batch['input_ids'][:, -max_length:]
                else:
                    x = input_ids[:, :max_length]
                    y = input_ids[:, 1:max_length+1]
                    x_mask = attention_mask[:, :max_length]
                input_ids = input_ids[:, max_length:]
                attention_mask = attention_mask[:, max_length:]
                xBatch, yBatch = x.to(device), y.to(device)
                xMaskBatch = x_mask.to(device)
                _, loss = model(xBatch, xMaskBatch, yBatch)
                losses[k] = loss.item()
                k += 1
        out[split] = losses.mean()
    model.train()
    return out


def top_k_top_p_filter(logits, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        # shift right by 1 since filter includes the first index that exceeds top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0

        # convert to original indexing
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


class Tokenizer:
	def __init__(self, tokenizer_file):
		with open(tokenizer_file, 'r') as f:
			data = json.load(f)

		self._char_to_idx = data['char_to_idx']
		self._idx_to_char = data['idx_to_char']
	
	def encode(self, text):
		return [self._char_to_idx[c] for c in text]
	
	def decode(self, input_ids):
		return ''.join([self._idx_to_char[str(i)] for i in input_ids])