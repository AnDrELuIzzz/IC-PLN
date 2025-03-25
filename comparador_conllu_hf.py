from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from conllu import parse_incr
from seqeval.metrics import classification_report
from tabulate import tabulate
import os

# ==========================================
# 1. Função para Parsear Arquivo CONLL-U
# ==========================================
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

# ==========================================
# 2. Carregar Modelo BERTimbau
# ==========================================
def load_hf_models():
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

# ==========================================
# 3. Alinhamento de Previsões (Corrigido)
# ==========================================
def align_predictions(text, gold_tokens, preds):
    aligned = []
    current_pos = 0
    
    for token in gold_tokens:
        start = text.find(token, current_pos)
        end = start + len(token)
        current_pos = end
        
        best_match = None
        for pred in preds:
            if (start >= pred['start'] and end <= pred['end']) or \
               (pred['start'] <= start <= pred['end']) or \
               (pred['start'] <= end <= pred['end']):
                
                if not best_match or (pred['end'] - pred['start']) < (best_match['end'] - best_match['start']):
                    best_match = pred
                    
        aligned.append(best_match['entity_group'] if best_match else "UNK")
    
    return aligned

# ==========================================
# 4. Avaliação (Corrigida)
# ==========================================
def evaluate_hf(sentences, nlp):
    results = []
    for sent in sentences:
        text = sent['text']
        preds = nlp(text)
        
        # Gerar spans dos tokens preditos
        tokens_hf = nlp.tokenizer.tokenize(text)
        encoding = nlp.tokenizer(text, return_offsets_mapping=True)
        pred_spans = encoding['offset_mapping'][1:-1]  # Ignora [CLS] e [SEP]
        
        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'pred_tokens': tokens_hf,
            'gold_pos': sent['gold_pos'],
            'pred_pos': align_predictions(text, sent['tokens'], preds),
            'gold_heads': sent['gold_heads'],
            'pred_heads': [0]*len(sent['tokens']),
            'gold_deprels': sent['gold_deprels'],
            'pred_deprels': ['dep']*len(sent['tokens']),
            'gold_lemmas': sent['gold_lemmas'],
            'pred_lemmas': ['lemma']*len(sent['tokens']),
            'gold_spans': sent['gold_spans'],
            'pred_spans': pred_spans  # Adicionado
        })
    return results

# ==========================================
# 5. Cálculo de Métricas (Corrigido)
# ==========================================
def calculate_metrics(results):
    total_tokens = pos_correct = 0
    tp_token = fp_token = fn_token = 0
    
    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        
        # Acurácia POS
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        
        # Tokenização
        gold_spans = set(sent['gold_spans'])
        pred_spans = set(tuple(span) for span in sent['pred_spans'])
        
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
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# ==========================================
# 6. Saída dos Resultados (Corrigido)
# ==========================================
def save_results_to_file(results, metrics, output_file, max_sentences=5):
    metric_rows = [
        ["Acurácia de POS", f"{metrics['pos_accuracy']:.2%}"],
        ["Precisão em Tokenização", f"{metrics['token_precision']:.2%}"],
        ["Recall em Tokenização", f"{metrics['token_recall']:.2%}"],
        ["F1-Score em Tokenização", f"{metrics['token_f1']:.2%}"]
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Métricas Gerais ===\n")
        f.write(tabulate(metric_rows, headers=["Métrica", "Valor"], tablefmt="grid") + "\n\n")
        
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write(f"Texto: {sent['text']}\n\n")
            
            # Tabela de Tokens (Corrigida)
            token_table = tabulate(
                [[sent['gold_tokens'], sent['pred_tokens']]],  # ← Colchete corrigido
                headers=["Gold Tokens", "HF Tokens"],
                tablefmt="grid"
            )
            f.write(token_table + "\n\n")
            
            # Tabela Comparativa
            comp_rows = []
            for j in range(len(sent['gold_tokens'])):
                comp_rows.append([
                    sent['gold_tokens'][j],
                    sent['gold_pos'][j],
                    sent['pred_pos'][j],
                    sent['gold_heads'][j],
                    sent['pred_heads'][j],
                    sent['gold_deprels'][j],
                    sent['pred_deprels'][j]
                ])
                
            f.write(tabulate(
                comp_rows,
                headers=["Token", "Gold POS", "HF POS", "Gold HEAD", "HF HEAD", "Gold DEP", "HF DEP"],
                tablefmt="grid"
            ) + "\n")

# ==========================================
# 7. Execução Principal
# ==========================================
if __name__ == "__main__":
    # Configurações
    txt_dir = "analise_hf2"
    os.makedirs(txt_dir, exist_ok=True)
    
    # Processamento
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    nlp = load_hf_models()
    results = evaluate_hf(sentences, nlp)
    metrics = calculate_metrics(results)
    
    # Salvar resultados
    save_results_to_file(
        results,
        metrics,
        output_file=os.path.join(txt_dir, "resultados_completos.txt")
    )
    
    print("✅ Processamento concluído!")
    print(f"Resultados em: '{txt_dir}/resultados_completos.txt'")