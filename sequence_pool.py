from sequence import Sequence
import torch
import numpy as np
import threading
import time
from collections import deque


class SequencePool:

    def __init__(self, vocab_size=None, min_seq_len=None, max_seq_len=None, truncate_to_max_seq_len=False,
                 shard_path_pattern=None, num_shards=None, token_dtype=np.uint16,
                 start_id=50256, end_id=50256, min_tokens_threshold=None):
        """
        A thread-safe pool of training sequences with optional background shard loading.

        Sequences are consumed FIFO via get_sequences(). Once returned, the pool no longer
        holds references to them — the caller owns them and memory is freed when the
        caller drops them.

        Args:
            vocab_size: Maximum vocab size; sequences with out-of-range tokens are discarded.
            min_seq_len: Discard sequences shorter than this.
            max_seq_len: Discard (or truncate) sequences longer than this.
            truncate_to_max_seq_len: If True, truncate long sequences instead of discarding.
            shard_path_pattern: Format string for shard paths with {shard_index}, e.g.
                                "fineweb10B/fineweb_train_{shard_index:06d}.bin".
                                Required for background loading.
            num_shards: Total number of shards (1-indexed). Required for background loading.
            token_dtype: Numpy dtype for reading raw tokens from shard files.
            start_id: Token ID marking the start of a sequence.
            end_id: Token ID marking the end of a sequence.
            min_tokens_threshold: When available tokens drop below this, the background
                                  loader will load the next shard. None disables background loading.
        """
        # Sequence storage
        self._queue = deque()
        self._tokens_available = 0
        self._next_seq_id = 0

        # Filtering config
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.truncate_to_max_seq_len = truncate_to_max_seq_len

        # Shard / loading config
        self.shard_path_pattern = shard_path_pattern
        self.num_shards = num_shards
        self.token_dtype = token_dtype
        self.start_id = start_id
        self.end_id = end_id
        self.min_tokens_threshold = min_tokens_threshold
        self._next_shard_index = 1  # 1-indexed
        self._all_shards_loaded = False

        # Synchronization
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._load_event = threading.Event()

        # Start background loader if configured
        self._loader_thread = None
        if self.min_tokens_threshold is not None and self.shard_path_pattern is not None:
            self._loader_thread = threading.Thread(target=self._loader_loop, daemon=True)
            self._loader_thread.start()

    # ------------------------------------------------------------------ #
    #  Background loader                                                  #
    # ------------------------------------------------------------------ #

    def _loader_loop(self):
        while not self._stop_event.is_set():
            self._load_event.wait(timeout=0.5)
            self._load_event.clear()

            if self._stop_event.is_set():
                break

            # Load shards until we're above threshold or exhausted
            while not self._stop_event.is_set():
                with self._lock:
                    above_threshold = self._tokens_available >= self.min_tokens_threshold
                    exhausted = self._all_shards_loaded
                if above_threshold or exhausted:
                    break
                self._load_next_shard()

    def _load_next_shard(self):
        with self._lock:
            if self._next_shard_index > self.num_shards:
                self._all_shards_loaded = True
                self._data_available.notify_all()
                return 0
            shard_index = self._next_shard_index
            self._next_shard_index += 1

        shard_path = self.shard_path_pattern.format(shard_index=shard_index)

        # Parse outside the lock (this is the expensive I/O + numpy work)
        new_seqs, new_tokens = self._parse_shard(shard_path)

        with self._lock:
            base_id = self._next_seq_id
            for i, seq in enumerate(new_seqs):
                seq.seq_id = base_id + i
            self._next_seq_id = base_id + len(new_seqs)
            self._queue.extend(new_seqs)
            self._tokens_available += new_tokens
            if shard_index >= self.num_shards:
                self._all_shards_loaded = True
            self._data_available.notify_all()

        print(f"[Loader] Loaded {len(new_seqs)} sequences ({new_tokens} tokens) from {shard_path}", flush=True)
        return len(new_seqs)

    # ------------------------------------------------------------------ #
    #  Shard parsing (pure — no shared state mutation)                    #
    # ------------------------------------------------------------------ #

    def _parse_shard(self, shard_path):
        tokens_np = np.fromfile(shard_path, dtype=self.token_dtype)

        start_inds = np.argwhere(tokens_np == self.start_id).reshape(-1)
        end_inds = np.argwhere(tokens_np == self.end_id).reshape(-1)

        if self.start_id == self.end_id:
            end_inds = end_inds[1:]

        num_seqs = min(len(start_inds), len(end_inds))

        new_seqs = []
        total_tokens = 0
        for i in range(num_seqs):
            si, ei = int(start_inds[i]), int(end_inds[i])
            if si >= ei:
                continue

            inp_np = tokens_np[si:ei]
            tgt_np = np.append(inp_np[1:], self.end_id)

            inp_tokens = torch.from_numpy(inp_np.copy()).long()
            targets = torch.from_numpy(tgt_np.copy()).long()

            seq_len = len(inp_tokens)
            if self.min_seq_len is not None and seq_len < self.min_seq_len:
                continue
            if self.max_seq_len is not None and seq_len > self.max_seq_len:
                if self.truncate_to_max_seq_len:
                    inp_tokens = inp_tokens[:self.max_seq_len]
                    targets = targets[:self.max_seq_len]
                else:
                    continue
            if self.vocab_size is not None and (inp_tokens.max() >= self.vocab_size or targets.max() >= self.vocab_size):
                continue

            seq = Sequence(inp_tokens, targets=targets, seq_id=0)  # id assigned under lock later
            new_seqs.append(seq)
            total_tokens += len(seq)

        return new_seqs, total_tokens

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def tokens_available(self):
        with self._lock:
            return self._tokens_available

    @property
    def all_shards_loaded(self):
        with self._lock:
            return self._all_shards_loaded

    def load_sequences_from_shard(self, shard_path, token_dtype=np.uint16, start_id=50256, end_id=50256):
        """Synchronous shard loading (original API, still works)."""
        self.token_dtype = token_dtype
        self.start_id = start_id
        self.end_id = end_id
        new_seqs, new_tokens = self._parse_shard(shard_path)

        with self._lock:
            base_id = self._next_seq_id
            for i, seq in enumerate(new_seqs):
                seq.seq_id = base_id + i
            self._next_seq_id = base_id + len(new_seqs)
            self._queue.extend(new_seqs)
            self._tokens_available += new_tokens

        return len(new_seqs)

    def get_sequences(self, num_seqs=None, min_token_count=None, max_token_count=None):
        """
        Consume sequences from the pool (FIFO).

        Returned sequences are removed from the pool — the caller owns them.

        Provide one of:
            num_seqs: Return exactly this many sequences (or fewer if pool is exhausted).
            min/max_token_count: Return sequences whose total token count falls within range.
                If shards are still loading and we don't have min_token_count yet, blocks
                until enough data arrives or all shards are exhausted.
        """
        if min_token_count is None and max_token_count is None:
            if num_seqs is None:
                raise ValueError("Provide num_seqs or min/max_token_count")
            return self._get_by_count(num_seqs)
        return self._get_by_tokens(min_token_count, max_token_count)

    def add_random_sequences(self, num_seqs, seq_len, start_id=50256, end_id=50256):
        with self._lock:
            for _ in range(num_seqs):
                tokens = torch.randint(0, self.vocab_size, (seq_len,)).long()
                targets = torch.cat((tokens.clone()[1:], torch.tensor([end_id]).long()))
                seq = Sequence(tokens, targets=targets, seq_id=self._next_seq_id)
                self._queue.append(seq)
                self._tokens_available += len(seq)
                self._next_seq_id += 1

    def prefetch_initial_shards(self, num_shards=1):
        """Synchronously load the first N shards so training can start immediately."""
        for _ in range(num_shards):
            with self._lock:
                if self._all_shards_loaded:
                    break
            self._load_next_shard()

    def stop(self):
        """Stop the background loader thread."""
        self._stop_event.set()
        self._load_event.set()
        with self._lock:
            self._data_available.notify_all()
        if self._loader_thread is not None:
            self._loader_thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    #  Internal consumption helpers                                       #
    # ------------------------------------------------------------------ #

    def _get_by_count(self, num_seqs):
        result = []
        with self._lock:
            for _ in range(min(num_seqs, len(self._queue))):
                seq = self._queue.popleft()
                self._tokens_available -= len(seq)
                result.append(seq)
        self._load_event.set()
        return result

    def _get_by_tokens(self, min_token_count, max_token_count):
        """
        Collect sequences until token budget is met.
        Blocks if min_token_count can't be satisfied yet but more shards are pending.
        """
        with self._lock:
            while True:
                seqs, token_count = self._try_collect(min_token_count, max_token_count)

                if seqs is not None:
                    # Commit: pop collected sequences from the front of the deque
                    for _ in range(len(seqs)):
                        self._queue.popleft()
                    self._tokens_available -= token_count
                    self._load_event.set()
                    return seqs

                # Couldn't satisfy min_token_count — wait if more data might arrive
                if self._all_shards_loaded:
                    return []

                # Signal loader and block until new data is appended
                self._load_event.set()
                self._data_available.wait(timeout=1.0)

    def _try_collect(self, min_token_count, max_token_count):
        """
        Peek at the front of the deque and try to collect sequences that fit the
        token budget. Returns (list_of_seqs, total_tokens) on success, or (None, 0)
        if min_token_count can't be met from what's currently available.

        Must be called while holding self._lock.
        """
        collected = []
        total_tokens = 0

        for seq in self._queue:
            seq_len = len(seq)

            if max_token_count is not None and total_tokens + seq_len > max_token_count:
                break

            collected.append(seq)
            total_tokens += seq_len

        if min_token_count is not None and total_tokens < min_token_count:
            return None, 0

        return collected, total_tokens