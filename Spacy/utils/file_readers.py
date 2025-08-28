import os
from spacy.tokens import Doc

def parse_conllu(file_path):
    """
    Faz parsing de um arquivo no formato CONLL-U extraindo informações essenciais.
    Detalhes técnicos:
    - Identifica metadados como texto original e id da sentença.
    - A partir do token '1\t' inicia a extração dos tokens e atributos (POS, HEAD, DEPREL).
    - Calcula spans dos tokens de acordo com sua posição no texto original, utilizando 
      a função find() para manter consistência com a formatação do texto.
    """
    sentences = []
    current_sent = {}
    in_sentence = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# text ='):
                current_sent['text'] = line.split('=', 1)[1].strip()
            elif line.startswith('# sent_id ='):
                current_sent['sent_id'] = line.split('=', 1)[1].strip()
            elif line and line[0].isdigit() and '-' not in line.split('\t')[0]:
                if not in_sentence:
                    in_sentence = True
                    current_sent['tokens'] = []
                    current_sent['gold_pos'] = []
                    current_sent['gold_heads'] = []
                    current_sent['gold_deprels'] = []
                    current_sent['gold_lemmas'] = []  # Nova chave para armazenar lemas
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_pos'].append(parts[3])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
                current_sent['gold_lemmas'].append(parts[2])  # Captura o lema
            elif line == '':
                if in_sentence and current_sent.get('tokens'):
                    text = current_sent['text']
                    current_pos = 0
                    spans = []
                    for token in current_sent['tokens']:
                        start = text.find(token, current_pos)
                        end = start + len(token)
                        spans.append((start, end))
                        current_pos = end
                    current_sent['gold_spans'] = spans
                    sentences.append(current_sent)
                    current_sent = {}
                    in_sentence = False
    return sentences

def read_wikiner(path, max_sentences=None):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens, tags = [], []
            for tok in line.strip().split():
                try:
                    word, pos, ner = tok.split("|")
                except ValueError:
                    continue
                tokens.append(word)
                tags.append(ner)
            if tokens:
                sentences.append((tokens, tags))
            if max_sentences and i >= max_sentences:
                break
    return sentences

def get_gold_entities(tokens, tags):
    entities = []
    current_entity, current_label = [], None
    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [], None
        else:
            label = tag.split("-")[-1]
            if current_label == label:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((" ".join(current_entity), current_label))
                current_entity, current_label = [token], label
    if current_entity:
        entities.append((" ".join(current_entity), current_label))
    return entities


