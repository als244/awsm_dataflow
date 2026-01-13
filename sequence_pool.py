from sequence import Sequence
import torch
import numpy as np


class SequencePool:

    def __init__(self, vocab_size=None, min_seq_len=None, max_seq_len=None, truncate_to_max_seq_len=False):
        self.sequences = []
        self.vocab_size = vocab_size
        self.cur_seq_ind = 0
        self.next_seq_id = 0
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.trucate_to_max_seq_len = truncate_to_max_seq_len

     
    ## for 1-D discrete token ids for next token prediction
    def load_sequences_from_shard(self, shard_path, token_dtype=np.uint16, start_id=50256, end_id=50256):

        tokens_np = np.fromfile(shard_path, dtype=token_dtype)

        start_inds = np.argwhere(tokens_np == start_id).reshape(-1)
        end_inds = np.argwhere(tokens_np == end_id).reshape(-1)

        ## if start is same as end, then shift the end inds to right by one
        if start_id == end_id:
            end_inds = end_inds[1:]
        

        num_seqs = min(len(start_inds), len(end_inds))

        num_seqs_loaded = 0
        for i in range(num_seqs):
            start_ind = start_inds[i]
            end_ind = end_inds[i]

            if start_ind >= end_ind:
                print(f"Start index is greater than end index for sequence {i}")
                continue

            inp_tokens_np = tokens_np[start_ind:end_ind]
            targets_np = inp_tokens_np[1:]
            targets_np = np.append(targets_np, end_id)

            ## convert to torch
            inp_tokens = torch.from_numpy(inp_tokens_np).long()
            targets = torch.from_numpy(targets_np).long()

            if self.min_seq_len is not None and len(inp_tokens) < self.min_seq_len:
                continue
            if self.max_seq_len is not None and len(inp_tokens) > self.max_seq_len:
                if self.trucate_to_max_seq_len:
                    inp_tokens = inp_tokens[:self.max_seq_len]
                    targets = targets[:self.max_seq_len]
                else:
                    continue

            if self.vocab_size is not None and (inp_tokens.max() >= self.vocab_size or targets.max() >= self.vocab_size):
                continue

            seq = Sequence(inp_tokens, targets=targets, seq_id=self.next_seq_id)
            self.sequences.append(seq)
            num_seqs_loaded += 1
            self.next_seq_id += 1

        return num_seqs_loaded
    
    def get_sequences(self, num_seqs=None, min_token_count=None, max_token_count=None):
        
        
        if min_token_count is None and max_token_count is None:
            if num_seqs is None:
                raise ValueError("num_seqs must be provided if min_token_count and max_token_count are not provided")
            seqs = self.sequences[self.cur_seq_ind:self.cur_seq_ind + num_seqs]
            self.cur_seq_ind += num_seqs
            return seqs
        
        total_tokens = 0
        tmp_cur_seq_ind = self.cur_seq_ind
        if min_token_count is not None:    
            while total_tokens < min_token_count and tmp_cur_seq_ind < len(self.sequences):
                total_tokens += len(self.sequences[tmp_cur_seq_ind])
                tmp_cur_seq_ind += 1
            

        if max_token_count is not None:
            while total_tokens < max_token_count and tmp_cur_seq_ind < len(self.sequences):
                next_seq_tokens = len(self.sequences[tmp_cur_seq_ind])
                if total_tokens + next_seq_tokens > max_token_count:
                    break
                total_tokens += next_seq_tokens
                tmp_cur_seq_ind += 1

        if min_token_count is not None and total_tokens < min_token_count:
            return []
        if max_token_count is not None and total_tokens > max_token_count:
            return []
        
        seqs = self.sequences[self.cur_seq_ind:tmp_cur_seq_ind]
        self.cur_seq_ind = tmp_cur_seq_ind
        return seqs
    
    
    def add_random_sequences(self, num_seqs, seq_len, start_id=50256, end_id=50256):
        
        for s in range(num_seqs):
            tokens = torch.randint(0, self.vocab_size, (seq_len,)).long()
            targets = torch.cat((tokens.clone()[1:], torch.tensor([end_id]).long()))
            seq = Sequence(tokens, targets=targets, seq_id=self.next_seq_id)
            self.sequences.append(seq)
            self.next_seq_id += 1

