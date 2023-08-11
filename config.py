from dataclasses import dataclass

@dataclass
class GPTConfig:
	batch_size:int
	max_length:int
	lr:float
	n_steps:int
	eval_interval:int
	eval_iters:int
	n_embd:int
	n_head:int
	n_layer:int
	dropout:float
	vocab_size:int
	pad_token:int