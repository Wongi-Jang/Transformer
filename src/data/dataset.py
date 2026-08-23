import torch
from torch.utils.data import Dataset

PAD_ID = 0


class TranslationDataset(Dataset):
    def __init__(self, hf_split, src_tokenizer, tgt_tokenizer, src_lang="en", tgt_lang="de", max_len=128):
        self.data = hf_split
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        src_ids = self.src_tokenizer.encode(
            row[self.src_lang]).ids[:self.max_len]
        tgt_ids = self.tgt_tokenizer.encode(
            row[self.tgt_lang]).ids[:self.max_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, pad_id=PAD_ID):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_id)

    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    src_padding_mask = (src_padded == pad_id)
    tgt_padding_mask = (tgt_input == pad_id)

    return {
        "src": src_padded,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask
    }
