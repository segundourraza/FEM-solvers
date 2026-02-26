STABILIZATION_CONSTANT = 2.5
LEAVE_TQDM_BAR = True


try: 
    from tqdm import tqdm
except ModuleNotFoundError:
    import sys
    import time

    class tqdm:
        def __init__(self, iterable, desc=None, leave=True):
            self.iterable = iterable
            self.desc = desc or ""
            self.leave = leave
            self.total = len(iterable) if hasattr(iterable, "__len__") else None
            self.start_time = None

        def __iter__(self):
            self.start_time = time.time()
            for i, item in enumerate(self.iterable, 1):
                self._print_progress(i)
                yield item
            self._finish()

        def _print_progress(self, current):
            if self.total:
                percent = current / self.total
                bar_length = 30
                filled = int(bar_length * percent)
                bar = "#" * filled + "-" * (bar_length - filled)
                elapsed = time.time() - self.start_time
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (self.total - current) / rate if rate > 0 else 0

                msg = (
                    f"\r{self.desc} |{bar}| "
                    f"{current}/{self.total} "
                    f"[{elapsed:0.1f}s<{remaining:0.1f}s]"
                )
            else:
                msg = f"\r{self.desc} {current}"

            sys.stdout.write(msg)
            sys.stdout.flush()

        def _finish(self):
            if self.leave:
                sys.stdout.write("\n")
            else:
                sys.stdout.write("\r")
            sys.stdout.flush()
        
        def close(self):
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        def write(str):
            sys.stdout.write("\n" + str)
            sys.stdout.flush()
        
        

def _progress_range(iterable, desc=None):
    return tqdm(
        iterable,
        desc=desc,
        leave=LEAVE_TQDM_BAR
    )
