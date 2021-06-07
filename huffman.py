# %%

from collections import deque
from itertools import product
from functools import reduce
from typing import Iterable, Optional, Sequence
import operator


class Node:
    '''Huffman Tree Node

    When token is None, then node is an internal node
    '''
    token: Optional[str]
    prob: float
    left: Optional['Node']
    right: Optional['Node']

    def __init__(self, token: Optional[str], prob: float, left: Optional['Node'] = None, right: Optional['Node'] = None) -> None:
        self.token = token
        self.prob = prob
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f'Node(token={self.token},prob={self.prob}'


class HuffTree:
    '''Huffman Tree that is built upon initialization'''

    root: Node

    def __init__(self, tokens: list[str], prob: list[float], sorted: bool = False) -> None:
        assert len(tokens) == len(
            prob), 'len of tokens should match len of frequencies'
        pairs = list(zip(prob, tokens))  # e.g. (3, 'a')

        # sort tokens according to frequency, and then use linear-time algo
        if not sorted:
            # heapq
            pairs.sort()

        # else:
        q0 = deque(Node(prob=prob, token=token) for prob, token in pairs)
        q1: deque[Node] = deque()
        while len(q1) + len(q0) > 1:

            front: list[Node] = []

            for _ in range(2):
                if len(q1) == 0:
                    front.append(q0.popleft())
                elif len(q0) == 0:
                    front.append(q1.popleft())
                else:
                    if q0[0].prob < q1[0].prob:
                        front.append(q0.popleft())
                    else:
                        front.append(q1.popleft())
            q1.append(Node(prob=front[0].prob + front[1].prob,
                           token=None, left=front[0], right=front[1]))

        if len(q1) == 1:
            self.root = q1.popleft()
        else:
            raise ValueError("You have only 1 value to encode")

    def get_info(self) -> list[tuple[str, str, float]]:
        '''Walk through the tree and extract info as `[(code, token, prob),...]`'''
        front: list[tuple[str, Node]] = [('', self.root)]
        out: list[tuple[str, str, float]] = []
        while front:
            code, node = front.pop()
            # right first here since we're on a stack
            if node.right is not None:
                front.append((code + '1', node.right))
            if node.left is not None:
                front.append((code + '0', node.left))
            else:
                assert node.token is not None
                out.append((code, node.token, node.prob))
        return out

    def decode(self, bin_str: str) -> str:
        '''Decode a string with given Huffman Tree'''
        chars: list[str] = []
        pos = self.root
        for c in bin_str:
            if c == '0':
                if pos.left is None:
                    raise ValueError('Invalid binary code')
                else:
                    pos = pos.left
            elif c == '1':
                if pos.right is None:
                    raise ValueError('Invalid binary code')
                else:
                    pos = pos.right

            # convetion from geekforgeeks, use '$' to denote internal node
            if pos.token is None:
                chars.append(str(pos.token))
                pos = self.root

        return ''.join(chars)


def shannon_entropy(probs: Iterable[float]) -> float:
    '''Calculates the theoretical entropy for the given Iterable of probabilities'''
    from math import log2
    return -sum(p * log2(p) for p in probs)


def bpt(prob: Iterable[float], bin_code: Iterable[str], group_len: int) -> float:
    '''Takes in an Iterable of probabilities and an Iterable of their corresponding binary codes, and calculates the average bit per token'''
    return sum(a * len(b) for a, b in zip(prob, bin_code)) / group_len


def batch_to_len(tokens: Iterable[str], prob: Iterable[float], target_len: int) -> tuple[list[str], list[float]]:
    '''Batch together tokens together up to size of `len`. Returns all possible batches and their probabilities'''
    return list(map(''.join, product(tokens, repeat=target_len))), list(map(lambda p: reduce(operator.mul, p, 1), product(prob, repeat=target_len)))


def run(tokens: Sequence[str], probs: Iterable[float], target_len: int, sort_results: bool = True, limit_results: Optional[int] = 100):
    '''Run through the Huffman Algorithm to generate code for a given sequence of `tokens` and their `probabilities`, and how many tokens to bundle together.

    `target_len`: how many `tokens` to bundle together.
    `sort_results`: if false, don't sort the results with code. Retain the order from walking along the tree.
    `limit_results`: print up to this number of results. Print all if `None`.
    '''

    if target_len > 14:
        ans = input(
            f'You asked for target length of {target_len}. Make sure you have LOTS OF RAM or this may fail. Are you sure you want to continue? (y/N) ')
        if ans.strip().lower() not in {'y', 'yes'}:
            return

    print('=' * 20)
    print(f'tokens: {list(tokens)}')
    print(f'probs:{probs}')
    print(f'{len(tokens)**target_len} of values needed for repr.')
    print(f'Shannon entropy: {shannon_entropy(probs)}.')
    print('=' * 20)
    print('Generating all possible values...')

    values, new_probs = batch_to_len(tokens, probs, target_len)

    print('Generating Huffman codes...')
    huff_tree = HuffTree(values, new_probs)

    info = huff_tree.get_info()

    if sort_results:
        info.sort(key=lambda x: len(x[0]))

    print('Expected bits per token: ', end='')
    codes, values, end_probs = zip(*info)
    print(bpt(end_probs, codes, target_len))

    print(f'{len(info)} results.')
    if limit_results is None or limit_results > 1:
        if limit_results is not None:
            print(f'Printing up to {limit_results} results: ')
        else:
            print(f'Printing all results:')
        print('code\tvalue\tprob')
        for code, value, prob in info[:limit_results]:
            print(f'{code}\t{value}\t{prob}')


# Mr. Brooks: to produce HW output:
# # 1.
# run('abcde', [1/5]*5, 1)
# run('abcde', [1/5]*5, 7, limit_results=None)
# # 2.
# run(tokens='htm', probs=[1/4, 1/4, 1/2], target_len=1)
# # 3
run(tokens=['a','b','c','d','e'], probs=[1/3] + [1/6]*4, target_len=1)
# run(tokens='abcde', probs=[1/3] + [1/6]*4, target_len=5)


# This code took all my RAM and still didn't work. I have 16GB. Didn't work on COLAB with 12GB RAM either. Failed even when trying to generate all the possible "combinations" of bundles.
# run(tokens='abc', probs=[1/5]*3, target_len=20)
