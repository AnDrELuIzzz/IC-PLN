from transformers import AutoTokenizer, AutoModel
from udify import UdifyModel
from tabulate import tabulate
import os

def parse_conllu(file_path):
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
                    current_sent['gold_lemmas'] = []
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])
                current_sent['gold_pos'].append(parts[3])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
                current_sent['gold_lemmas'].append(parts[2])
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

def evaluate_huggingface(sentences, model_name="udify/udify-bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = UdifyModel.from_pretrained(model_name)
    
    results = []
    for sent in sentences:
        tokens = sent['tokens']
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        
        # Processar POS tags
        pos_ids = outputs['pos_tags'].argmax(-1).squeeze().tolist()
        word_ids = inputs.word_ids()
        pos_tags = []
        current_word = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != current_word:
                current_word = word_id
                pos_tags.append(model.config.id2pos[pos_ids[i]])
        
        # Processar dependências
        heads = outputs['head_indices'].squeeze().tolist()
        deprel_ids = outputs['head_tags'].argmax(-1).squeeze().tolist()
        deprels = [model.config.id2label[deprel_id] for deprel_id in deprel_ids]
        
        # Ajustar índices dos heads
        adjusted_heads = []
        for head in heads[:len(tokens)]:
            adjusted_heads.append(head if head <= len(tokens) else 0)
        
        # Lematização (simplificado)
        lemmas = outputs['lemmas'].argmax(-1).squeeze().tolist()
        pred_lemmas = [tokenizer.decode([lemma]) for lemma in lemmas[:len(tokens)]]
        
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': sent['text'],
            'gold_tokens': tokens,
            'gold_spans': sent['gold_spans'],
            'pred_tokens': tokens,
            'pred_spans': sent['gold_spans'],
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'gold_lemmas': sent['gold_lemmas'],
            'pred_pos': pos_tags,
            'pred_heads': adjusted_heads,
            'pred_deprels': deprels[:len(tokens)],
            'pred_lemmas': pred_lemmas
        })
    return results

def calculate_metrics(results):
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    lemma_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0
    
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        
        # Acurácia POS
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        
        # Acurácia Lemas
        lemma_correct += sum(1 for g, p in zip(sent['gold_lemmas'], sent['pred_lemmas']) if g == p)
        
        # UAS e LAS
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
        
        # Tokenização
        gold_spans = set(sent['gold_spans'])
        pred_spans = set(sent['pred_spans'])
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        tp_token += tp
        fp_token += fp
        fn_token += fn
    
    precision = tp_token / (tp_token + fp_token) if (tp_token + fp_token) > 0 else 0
    recall = tp_token / (tp_token + fn_token) if (tp_token + fn_token) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'pos_accuracy': pos_correct / total_tokens,
        'lemma_accuracy': lemma_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# As funções save_results_to_file e analyze_errors permanecem similares ao original

if __name__ == "__main__":
    txt_dir = "analise_transformers"
    os.makedirs(txt_dir, exist_ok=True)

    sentences = parse_conllu("seu_arquivo.conllu")
    results = evaluate_huggingface(sentences)
    metrics = calculate_metrics(results)
    
    # Salvar resultados
    with open(f"{txt_dir}/resultados.txt", "w") as f:
        f.write(f"Acurácia POS: {metrics['pos_accuracy']:.2%}\n")
        f.write(f"Acurácia Lemas: {metrics['lemma_accuracy']:.2%}\n")
        f.write(f"UAS: {metrics['uas']:.2%}\n")
        f.write(f"LAS: {metrics['las']:.2%}\n")

    print("Processamento concluído com Hugging Face Transformers!")