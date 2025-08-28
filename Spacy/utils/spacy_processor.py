import spacy
from spacy.tokens import Doc

def load_spacy_model(model_name):
    return spacy.load(model_name)

def process_conllu_sentence(sent, nlp):
    text = sent["text"]
    doc_raw = nlp(text)
    pred_spans = [(token.idx, token.idx + len(token)) for token in doc_raw]
    doc_gold = Doc(nlp.vocab, words=sent["tokens"])
    doc_gold = nlp(doc_gold)
    return {
        "sent_id": sent.get("sent_id", ""),
        "text": text,
        "gold_tokens": sent["tokens"],
        "gold_spans": sent["gold_spans"],
        "pred_tokens": [token.text for token in doc_raw],
        "pred_spans": pred_spans,
        "gold_pos": sent["gold_pos"],
        "gold_heads": sent["gold_heads"],
        "gold_deprels": sent["gold_deprels"],
        "gold_lemmas": sent["gold_lemmas"],
        "pred_pos": [token.pos_ for token in doc_gold],
        "pred_heads": [
            token.head.i + 1 if token.head != token else 0 for token in doc_gold
        ],
        "pred_deprels": [token.dep_ for token in doc_gold],
        "pred_lemmas": [token.lemma_ for token in doc_gold],
    }

def process_wikiner_sentence(tokens, nlp):
    text = " ".join(tokens)
    doc = nlp(text)
    pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
    return text, pred_ents


