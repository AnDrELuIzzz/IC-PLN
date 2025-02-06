import spacy
from spacy.tokens import Doc

def parse_conllu(file_path):
    """
    Parseia um arquivo CONLL-U, lidando com multi-word tokens e metadados.
    """
    sentences = []
    current_sent = {}
    in_sentence = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Metadados (ex: # text = ...)
            if line.startswith('# text ='):
                current_sent['text'] = line.split('=', 1)[1].strip()
            elif line.startswith('# sent_id ='):
                current_sent['sent_id'] = line.split('=', 1)[1].strip()
            
            # Iniciar nova sentença ao encontrar o primeiro token (ID=1)
            elif line.startswith('1\t'):
                in_sentence = True
                current_sent['tokens'] = []
                current_sent['gold_pos'] = []
                current_sent['gold_heads'] = []
                current_sent['gold_deprels'] = []
            
            # Finalizar sentença ao encontrar linha vazia
            elif line == '':
                if in_sentence and current_sent.get('tokens'):
                    sentences.append(current_sent)
                    current_sent = {}
                    in_sentence = False
            
            # Processar tokens (ignorar multi-word tokens, ex: 2-3)
            elif in_sentence and line[0].isdigit() and '-' not in line.split('\t')[0]:
                parts = line.split('\t')
                current_sent['tokens'].append(parts[1])          # FORM
                current_sent['gold_pos'].append(parts[3])        # UPOS
                current_sent['gold_heads'].append(int(parts[6])) # HEAD
                current_sent['gold_deprels'].append(parts[7])    # DEPREL

    return sentences

#POS Tagging (Classificação Gramatical) Calcula a acurácia de POS (porcentagem de acertos).
#Dependency Parsing (Análise Sintática): Métricas: UAS (Unlabeled Attachment Score): Acertos na identificação do "pai" do token na árvore. LAS (Labeled Attachment Score): Acertos no "pai" e no tipo de relação (ex: nsubj, obj).


def evaluate_spacy(sentences, nlp_model):
    """
    Processa as sentenças com spaCy e retorna previsões alinhadas ao gold standard.
    """
    nlp = spacy.load(nlp_model)
    results = []

    for sent in sentences:
        # Criar Doc do spaCy com tokens alinhados
        doc = Doc(nlp.vocab, words=sent['tokens'])
        doc = nlp(doc)

        # Extrair previsões
        pred_pos = [token.pos_ for token in doc]
        pred_heads = [token.head.i + 1 if token.head != token else 0 for token in doc]  # Converter para índice 1-based
        pred_deprels = [token.dep_ for token in doc]

        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': sent.get('text', ''),
            'tokens': sent['tokens'],
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'pred_pos': pred_pos,
            'pred_heads': pred_heads,
            'pred_deprels': pred_deprels
        })

    return results

def calculate_metrics(results):
    """
    Calcula métricas de POS accuracy, UAS e LAS.
    """
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0

    for sent in results:
        n = len(sent['tokens'])
        total_tokens += n

        # Acurácia de POS
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)

        # UAS e LAS
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], 
            sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1

    return {
        'pos_accuracy': pos_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens
    }

def print_detailed_comparison(results, max_sentences=3):
    """
    Imprime comparações detalhadas para as primeiras 'max_sentences' sentenças.
    """
    for i, sent in enumerate(results[:max_sentences]):
        print(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===")
        print(f"Texto: {sent['text']}")
        print("\nToken | Gold POS | spaCy POS | Gold HEAD | spaCy HEAD | Gold DEPREL | spaCy DEPREL")
        for j in range(len(sent['tokens'])):
            print(
                f"{sent['tokens'][j]:<8} | "
                f"{sent['gold_pos'][j]:<8} | "
                f"{sent['pred_pos'][j]:<9} | "
                f"{sent['gold_heads'][j]:<9} | "
                f"{sent['pred_heads'][j]:<10} | "
                f"{sent['gold_deprels'][j]:<10} | "
                f"{sent['pred_deprels'][j]}"
            )

def save_results_to_file(results, metrics, output_file="resultados_completos.txt", max_sentences=None):
    """
    Salva métricas e comparações detalhadas em um arquivo de texto.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Escrever métricas gerais
        f.write("=== Métricas Gerais ===\n")
        f.write(f"Acurácia de POS: {metrics['pos_accuracy']:.2%}\n")
        f.write(f"UAS (Unlabeled Attachment Score): {metrics['uas']:.2%}\n")
        f.write(f"LAS (Labeled Attachment Score): {metrics['las']:.2%}\n\n")
        
        # Escrever comparações detalhadas
        if max_sentences is None:
            max_sentences = len(results)
            
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write(f"Texto: {sent['text']}\n")
            f.write("\nToken        | Gold POS  | spaCy POS | Gold HEAD | spaCy HEAD | Gold DEPREL   | spaCy DEPREL\n")
            f.write("------------|-----------|-----------|-----------|------------|---------------|-------------\n")
            
            for j in range(len(sent['tokens'])):
                line = (
                    f"{sent['tokens'][j]:<12} | "
                    f"{sent['gold_pos'][j]:<9} | "
                    f"{sent['pred_pos'][j]:<9} | "
                    f"{sent['gold_heads'][j]:<9} | "
                    f"{sent['pred_heads'][j]:<10} | "
                    f"{sent['gold_deprels'][j]:<13} | "
                    f"{sent['pred_deprels'][j]}"
                )
                f.write(line + "\n")

# Execução principal
if __name__ == "__main__":
    # 1. Parsear o arquivo
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    
    # 2. Processar com spaCy
    results = evaluate_spacy(sentences, "pt_core_news_lg")
    
    # 3. Calcular métricas
    metrics = calculate_metrics(results)
    
    # 4. Salvar resultados em arquivo
    save_results_to_file(results, metrics, max_sentences=len(results))  # Todas as sentenças
    
    # 5. Mostrar confirmação
    print("Processamento concluído! Resultados salvos em 'resultados_completos.txt'")
