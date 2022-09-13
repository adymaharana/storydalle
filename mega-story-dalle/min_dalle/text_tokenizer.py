from math import inf
from typing import List, Tuple
from emoji import demojize

class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))
        self.new_tokens = []

    def add_tokens(self, new_tokens):

        self.new_tokens = new_tokens
        original_length = len(self.token_from_subword)
        for t in new_tokens:
            self.token_from_subword[t] = len(self.token_from_subword)
        print("Increased vocabulary from %s to %s" % (original_length, len(self.token_from_subword)))

    def tokenize(self, text: str, is_verbose: bool = False) -> List[int]:
        sep_token = self.token_from_subword['</s>']
        cls_token = self.token_from_subword['<s>']
        unk_token = self.token_from_subword['<unk>']
        text = demojize(text, delimiters=['', ''])
        text = text.lower().encode("ascii", errors="ignore").decode()

        # tokens = [
        #     self.token_from_subword.get(subword, unk_token)
        #     for word in text.split(" ") if len(word) > 0
        #     for subword in self.get_byte_pair_encoding(word, is_verbose)
        # ]
        # print([subword for word in text.split(" ") if len(word) > 0
        #     for subword in self.get_byte_pair_encoding(word, is_verbose)])

        sub_words = []
        tokens = []
        for word in text.split(" "):
            if len(word) > 0:
                if word in self.new_tokens:
                    tokens.append(self.token_from_subword[word])
                    sub_words.append(word)
                else:
                    for subword in self.get_byte_pair_encoding(word, is_verbose):
                        tokens.append(self.token_from_subword.get(subword, unk_token))
                        sub_words.append(subword)
        # print(sub_words)

        return [cls_token] + tokens + [sep_token]

    def get_byte_pair_encoding(self, word: str, is_verbose: bool) -> List[str]:
        def get_pair_rank(pair: Tuple[str, str]) -> int:
            return self.rank_from_pair.get(pair, inf)

        subwords = [chr(ord(" ") + 256)] + list(word)
        while len(subwords) > 1:
            pairs = list(zip(subwords[:-1], subwords[1:]))
            pair_to_merge = min(pairs, key=get_pair_rank)
            if pair_to_merge not in self.rank_from_pair: break
            i = pairs.index(pair_to_merge)
            subwords = (
                (subwords[:i] if i > 0 else []) + 
                [subwords[i] + subwords[i + 1]] + 
                (subwords[i + 2:] if i + 2 < len(subwords) else [])
            )

        if is_verbose: print(subwords)
        return subwords