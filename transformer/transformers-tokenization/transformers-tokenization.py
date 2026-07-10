from typing import List, Dict


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {}
        self.id_to_word = {}
    
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
    
        for token in special_tokens:
            token_id = len(self.word_to_id)
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
    
        unique_words = set()
    
        for text in texts:
            for word in text.lower().split():
                unique_words.add(word)
    
        for word in sorted(unique_words):
            word_id = len(self.word_to_id)
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
    
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        ids = []
        unk_id = self.word_to_id[self.unk_token]
    
        for word in text.lower().split():
            ids.append(self.word_to_id.get(word, unk_id))
    
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        words = []

        for token_id in ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)

        return " ".join(words)