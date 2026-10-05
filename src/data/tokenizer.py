from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors,decoders
from datasets import load_dataset

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def train_tokenizer(texts, vocab_size=8000, save_path=None):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder=decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(single="<sos> $A <eos>",
                                                             special_tokens=[("<sos>", tokenizer.token_to_id("<sos>")), ("<eos>", tokenizer.token_to_id("<eos>"))
                                                                             ])
    if save_path:
        tokenizer.save(save_path)

    return tokenizer


def build_tokenizers(dataset, vocab_size=8000, out_dir="tokenizers"):
    Path(out_dir).mkdir(exist_ok=True)
    en_tok = train_tokenizer(
        dataset["train"]["en"], vocab_size, f"{out_dir}/en.json")
    de_tok = train_tokenizer(
        dataset["train"]["de"], vocab_size, f"{out_dir}/de.json")
    return en_tok, de_tok


if __name__ == "__main__":
    ds = load_dataset("bentrevett/multi30k")
    en_tok, de_tok = build_tokenizers(ds)
    out = en_tok.encode(ds["train"]["en"][0])
    print(out.tokens)
    print(out.ids)
