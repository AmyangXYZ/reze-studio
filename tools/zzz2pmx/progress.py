# -*- coding: utf-8 -*-
"""Stage timing, so a long build says where it is and what it cost."""
import sys
import time

_start = time.time()


class step:
    """`with step('the bodice'):` -- prints on entry, and the seconds on exit."""

    def __init__(self, what):
        self.what = what

    def __enter__(self):
        self.t = time.time()
        print(f'--- {self.what}', flush=True)
        return self

    def __exit__(self, *exc):
        print(f'    {self.what}: {time.time() - self.t:.1f}s'
              f'  (elapsed {time.time() - _start:.1f}s)', flush=True)
        return False


def done(path=''):
    print(f'{path} done in {time.time() - _start:.1f}s', flush=True)
