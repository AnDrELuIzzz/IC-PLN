import spacy  # Importa o spaCy para processamento de linguagem natural
from spacy.tokens import Doc  # Usado para criar um objeto Doc com tokens gold

# Função para parsear um arquivo CONLL-U e extrair anotações linguísticas
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
                # Comentário: Armazena o texto original da sentença
            elif line.startswith('# sent_id ='):
                current_sent['sent_id'] = line.split('=', 1)[1].strip()
                # Comentário: Armazena o identificador único da sentença
            
            # Inicia a captura de tokens assim que encontrar uma linha que comece com dígito (ignorando intervalos)
            elif line and line[0].isdigit() and '-' not in line.split('\t')[0]:
                if not in_sentence:
                    in_sentence = True
                    current_sent['tokens'] = []
                    current_sent['gold_pos'] = []
                    current_sent['gold_heads'] = []
                    current_sent['gold_deprels'] = []
                parts = line.split('\t')
                # Captura os atributos do token
                current_sent['tokens'].append(parts[1])
                current_sent['gold_pos'].append(parts[3])
                current_sent['gold_heads'].append(int(parts[6]))
                current_sent['gold_deprels'].append(parts[7])
            
            elif line == '':
                if in_sentence and current_sent.get('tokens'):
                    # Calcula os spans (posições iniciais e finais dos tokens) no texto original
                    text = current_sent['text']
                    current_pos = 0
                    spans = []
                    for token in current_sent['tokens']:
                        start = text.find(token, current_pos)
                        end = start + len(token)
                        spans.append((start, end))
                        current_pos = end
                        # Comentário: Localiza a posição do token garantindo que tokens repetidos sejam processados corretamente
                    current_sent['gold_spans'] = spans
                    sentences.append(current_sent)
                    current_sent = {}
                    in_sentence = False
                    # Comentário: Finaliza a sentença e reinicia as variáveis de controle
    
    return sentences

# Função para avaliar e comparar o processamento do spaCy com os dados gold
def evaluate_spacy(sentences, nlp_model):
    """
    Processa sentenças utilizando spaCy para realizar tokenização, POS tagging e extração de dependências.
    
    Detalhes técnicos:
    - Carrega o modelo spaCy especificado.
    - Processa o texto completo para gerar tokens "pred".
    - Alinha os tokens gold criando um objeto Doc e processa com o pipeline spaCy para obter atributos.
    - Extrai spans de tokens e mapeia as dependências levando em conta a indexação do spaCy.
    """
    nlp = spacy.load(nlp_model)
    results = []

    for sent in sentences:
        # Processa o texto original com o modelo spaCy para tokenização
        text = sent['text']
        doc_raw = nlp(text)
        pred_spans = [(token.idx, token.idx + len(token)) for token in doc_raw]
        # Comentário: Gera spans a partir dos índices dos tokens retornados pelo spaCy

        # Processa tokens gold criando um objeto Doc para alinhamento dos atributos linguísticos
        doc_gold = Doc(nlp.vocab, words=sent['tokens'])
        doc_gold = nlp(doc_gold)
        # Comentário: O processamento do doc_gold assegura que os atributos (POS, deprel) sejam compatíveis com o pipeline spaCy

        results.append({
            'sent_id': sent.get('sent_id', ''),
            'text': text,
            'gold_tokens': sent['tokens'],
            'gold_spans': sent['gold_spans'],
            'pred_tokens': [token.text for token in doc_raw],
            'pred_spans': pred_spans,
            'gold_pos': sent['gold_pos'],
            'gold_heads': sent['gold_heads'],
            'gold_deprels': sent['gold_deprels'],
            'pred_pos': [token.pos_ for token in doc_gold],
            'pred_heads': [token.head.i + 1 if token.head != token else 0 for token in doc_gold],
            'pred_deprels': [token.dep_ for token in doc_gold]
        })
        # Comentário: A estrutura dictionary mapeia os atributos gold e preditos para posterior comparação

    return results

# Função que calcula métricas de avaliação comparando dados de tokens, POS e dependências
def calculate_metrics(results):
    """
    Calcula métricas de comparação entre os dados gold e os preditos.
    
    Detalhes técnicos:
    - Acurácia de POS é calculada comparando elemento a elemento dos tokens.
    - UAS e LAS são derivados da verificação de acertos na estrutura de dependências.
    - Precision, Recall e F1 em tokenização são derivados dos spans sobrepostos.
    """
    total_tokens = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    tp_token, fp_token, fn_token = 0, 0, 0

    for sent in results:
        n = len(sent['gold_tokens'])
        total_tokens += n
        pos_correct += sum(1 for g, p in zip(sent['gold_pos'], sent['pred_pos']) if g == p)
        # Comentário: Compara POS tag a tag entre o gold e o predito
        
        for g_head, p_head, g_deprel, p_deprel in zip(
            sent['gold_heads'], sent['pred_heads'], sent['gold_deprels'], sent['pred_deprels']
        ):
            if g_head == p_head:
                uas_correct += 1
                if g_deprel == p_deprel:
                    las_correct += 1
                # Comentário: UAS e LAS são avaliados hierarquicamente, verificando dependências e relações
    
        # Avaliação de tokenização utilizando spans
        gold_spans = set(sent['gold_spans'])
        pred_spans = set(sent['pred_spans'])
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        tp_token += tp
        fp_token += fp
        fn_token += fn

    # Cálculo das métricas de precisão, recall e F1 para tokenização
    precision = tp_token / (tp_token + fp_token) if (tp_token + fp_token) > 0 else 0
    recall = tp_token / (tp_token + fn_token) if (tp_token + fn_token) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'pos_accuracy': pos_correct / total_tokens,
        'uas': uas_correct / total_tokens,
        'las': las_correct / total_tokens,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Função que salva os resultados e métricas em um arquivo de saída
def save_results_to_file(results, metrics, output_file="resultados_completos.txt", max_sentences=None):
    """
    Registra as métricas gerais e detalhes de cada sentença para análise posterior.
    
    Detalhes técnicos:
    - Abre o arquivo em modo escrita com encoding UTF-8 para garantir compatibilidade.
    - Escreve as métricas de forma formatada utilizando porcentagens para melhor visualização.
    - Permite limitar o número de sentenças detalhadas através do parâmetro max_sentences.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Métricas Gerais ===\n")
        f.write(f"Acurácia de POS: {metrics['pos_accuracy']:.2%}\n")
        f.write(f"UAS (Unlabeled Attachment Score): {metrics['uas']:.2%}\n")
        f.write(f"LAS (Labeled Attachment Score): {metrics['las']:.2%}\n")
        f.write(f"Precisão em Tokenização: {metrics['token_precision']:.2%}\n")
        f.write(f"Recall em Tokenização: {metrics['token_recall']:.2%}\n")
        f.write(f"F1-Score em Tokenização: {metrics['token_f1']:.2%}\n\n")
        
        if max_sentences is None:
            max_sentences = len(results)
            
        for i, sent in enumerate(results[:max_sentences]):
            f.write(f"\n=== Sentença {i+1} ({sent['sent_id']}) ===\n")
            f.write(f"Texto: {sent['text']}\n")
            
            # Exibe comparação entre tokens gold e preditos
            f.write("\nTokens Gold vs. spaCy:\n")
            f.write("Gold Tokens: " + " | ".join(sent['gold_tokens']) + "\n")
            f.write("spaCy Tokens: " + " | ".join(sent['pred_tokens']) + "\n")
            
            # Tabela comparativa de POS e dependências
            f.write("\nToken        | Gold POS  | spaCy POS | Gold HEAD | spaCy HEAD | Gold DEPREL   | spaCy DEPREL\n")
            f.write("------------|-----------|-----------|-----------|------------|---------------|-------------\n")
            for j in range(len(sent['gold_tokens'])):
                line = (
                    f"{sent['gold_tokens'][j]:<12} | "
                    f"{sent['gold_pos'][j]:<9} | "
                    f"{sent['pred_pos'][j]:<9} | "
                    f"{sent['gold_heads'][j]:<9} | "
                    f"{sent['pred_heads'][j]:<10} | "
                    f"{sent['gold_deprels'][j]:<13} | "
                    f"{sent['pred_deprels'][j]}"
                )
                f.write(line + "\n")
            # Comentário: Cada linha detalha as atribuições e permite identificar divergências na análise

# Execução principal
if __name__ == "__main__":
    
    # 1. Parsear o arquivo CONLL-U para extrair as sentenças e respectivos atributos
    sentences = parse_conllu("/home/andre/Dev-Ubuntu/IC/experimentos/scripts/data/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu")
    
    # 2. Processar cada sentença com o modelo spaCy para obter dados preditos
    results = evaluate_spacy(sentences, "pt_core_news_lg")
    
    # 3. Calcular as métricas comparativas entre os dados gold e os preditos
    metrics = calculate_metrics(results)
    
    # 4. Salvar os resultados detalhados e as métricas em um arquivo de saída
    save_results_to_file(results, metrics, max_sentences=len(results))
    
    # 5. Mensagem de confirmação da execução
    print("Processamento concluído! Resultados salvos em 'resultados_completos.txt'")
    