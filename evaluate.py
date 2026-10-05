import torch
from tokenizers import Tokenizer
from datasets import load_dataset
import sacrebleu
import argparse
from src.model.transformer import Transformer


def greedy_decode(model, src, max_len, device,sos_id,eos_id):
    # B=1
    model.eval()
    with torch.no_grad():
        memory, src_mask = model.encode(src.to(device))
        tgt = torch.tensor([[sos_id]], device=device)
        for _ in range(max_len):
            out = model.decode(tgt, memory, src_mask)  # (B, S_tgt, d_model)
            # (B, d_model) -> (B, tgt_vocab_size)
            logits = model.generator(out[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            tgt = torch.cat([tgt, next_token], dim=1)  # (B, S_tgt+1)
            if next_token.item() == eos_id:
                break
    return tgt.squeeze(0).tolist()  # (B,L) -> (L, )


def translate_sentence(model, sentence, src_tok, tgt_tok, device, max_len=128):
    src_ids = src_tok.encode(sentence).ids
    src = torch.tensor([src_ids])
    out_ids = greedy_decode(model, src, max_len, device,tgt_tok.token_to_id("<sos>"),tgt_tok.token_to_id("<eos>"))
    return tgt_tok.decode(out_ids, skip_special_tokens=True)


def evaluate_bleu(model, test_split, src_tok, tgt_tok, device, src_lang="en", tgt_lang="de"):
    hypotheses, references = [], []
    for row in test_split:
        hyp = translate_sentence(
            model, row[src_lang], src_tok, tgt_tok, device)
        hypotheses.append(hyp)
        references.append(row[tgt_lang])

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--post-ln", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    en_tok = Tokenizer.from_file("tokenizers/en.json")
    de_tok = Tokenizer.from_file("tokenizers/de.json")
    ds = load_dataset("bentrevett/multi30k")

    model = Transformer(src_vocab_size=en_tok.get_vocab_size(),
                        tgt_vocab_size=de_tok.get_vocab_size(),
                        d_model=512,pad_id=de_tok.token_to_id("<pad>"),norm_first=not args.post_ln).to(device)
    model.load_state_dict(torch.load(
        args.ckpt, map_location=device))

    score, hyps, refs = evaluate_bleu(
        model, ds["test"], en_tok, de_tok, device)
    print(f"BLEU score: {score:.2f}")

    for i in range(5):
        print(f"Source: {ds['test'][i]['en']}")
        print(f"Reference: {ds['test'][i]['de']}")
        print(f"Hypothesis: {hyps[i]}")
        print()


if __name__ == "__main__":
    main()
