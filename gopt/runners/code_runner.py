import logging
from time import time
from tqdm import tqdm
from pytimeparse import parse
from collections import defaultdict, deque

class CodeRunner:

    def __init__(self, max_iter, max_time, block_freq=1.0, max_cache_size=10):
        if isinstance(max_time, str):
            max_time = parse(max_time)

        def cache_creator():
            return deque(maxlen=max_cache_size)

        self.max_time = max_time
        self.max_iter = max_iter
        self.block_freq = block_freq
        self.start_time = None
        self.current_iter = 0
        self._timing_caches = defaultdict(cache_creator)
        self.max_cache_size = max_cache_size

    def start(self):
        self.start_time = time()
        self.progress_bar = tqdm(total=self.max_iter)

    def can_run_more(self):
        elapsed_time = time() - self.start_time

        if self.max_time is not None and elapsed_time >= self.max_time:
            logging.info(f'Out of time')
            return False

        if self.max_iter is not None and self.current_iter >= self.max_iter:
            logging.info(f'Reached max_iter')
            return False

        return True

    def get_best_block_size(self, history):

        # TODO we might want to improve that part of the code if it becomes
        # a significant bottleneck

        try:
            best_ratio = max(iters / time for (iters, time) in history)
            return int(best_ratio / self.block_freq)
        except:
            return 100


    def run_block(self, code, *args, **kwargs):

        if self.start_time is None:
            logger = logging.getLogger('gopt.code_runner')
            logger.info(f'Start of optimization')
            self.start()

        if not self.can_run_more():
            self.progress_bar.close()
            raise StopIteration

        cache = self._timing_caches[code]
        best_block_size = self.get_best_block_size(cache)

        start_code = time()
        result = code(*args, **kwargs, iterations=best_block_size)
        elapsed = time() - start_code

        self.current_iter += best_block_size
        self.progress_bar.update(best_block_size)
        self.progress_bar.set_postfix({'loss': result, 'bs':best_block_size})
        cache.append((best_block_size, elapsed))

        return result
