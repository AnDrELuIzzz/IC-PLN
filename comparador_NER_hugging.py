import torch
from transformers import pipeline
from sklearn.metrics import classification_report
from tabulate import tabulate
import os
import warnings

# Ignorar avisos comuns da Hugging Face
warnings.filterwarnings("ignore")

def read_wikiner(path, max_sentences=None):
    """Lê arquivo WikiNER no formato: palavra|pos|ner."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break
            tokens, tags = [], []
            # Usar re.split para lidar com múltiplos espaços ou linhas mal formatadas
            parts = [p for p in line.strip().split(' ') if p]
            for tok in parts:
                try:
                    word, pos, ner = tok.split("|")
                    tokens.append(word)
                    tags.append(ner)
                except ValueError:
                    # Ignora tokens malformados, como '||'
                    continue
            if tokens:
                sentences.append((tokens, tags))
    return sentences

def align_predictions_to_tokens(tokens, predictions):
    """
    Alinha as previsões do pipeline (que podem ser sub-tokens) aos tokens originais.
    Retorna uma lista de tags IOB do mesmo tamanho da lista de tokens.
    """
    if not predictions:
        return ["O"] * len(tokens)

    token_tags = ["O"] * len(tokens)
    text = " ".join(tokens)
    
    # Mapeia cada caractere do texto de volta ao seu índice de token original
    char_to_token_idx = []
    for i, token in enumerate(tokens):
        char_to_token_idx.extend([i] * (len(token) + 1)) # +1 para o espaço

    for pred in predictions:
        start, end = pred['start'], pred['end']
        
        # Garante que os índices não ultrapassem os limites do texto
        if start >= len(char_to_token_idx) or end >= len(char_to_token_idx):
            continue

        # Encontra os tokens que correspondem ao início e fim da entidade
        start_token_idx = char_to_token_idx[start]
        end_token_idx = char_to_token_idx[end - 1] # -1 porque 'end' é exclusivo

        label = pred['entity_group']
        
        # Aplica a tag IOB
        # A primeira tag é 'B-' (Beginning)
        token_tags[start_token_idx] = f"B-{label}"
        # As tags seguintes são 'I-' (Inside)
        for i in range(start_token_idx + 1, end_token_idx + 1):
            token_tags[i] = f"I-{label}"
            
    return token_tags

def evaluate_ner_model(ner_pipeline, sentences, max_sent=50):
    """
    Avalia o modelo NER usando alinhamento a nível de token (IOB).
    """
    all_true_tags = []
    all_pred_tags = []

    print(f"Processando {min(max_sent, len(sentences))} sentenças...")

    for i, (tokens, gold_tags) in enumerate(sentences[:max_sent], 1):
        if i % 10 == 0:
            print(f"  - Processando sentença {i}/{min(max_sent, len(sentences))}")

        text = " ".join(tokens)
        
        # O pipeline pode falhar em textos muito curtos ou complexos
        try:
            pred_entities = ner_pipeline(text)
        except Exception as e:
            print(f"Erro no pipeline na sentença {i}: {e}")
            pred_entities = []

        # Alinha as previsões aos tokens originais
        pred_tags = align_predictions_to_tokens(tokens, pred_entities)

        # Garante que as listas de tags tenham o mesmo tamanho
        if len(gold_tags) == len(pred_tags):
            all_true_tags.extend(gold_tags)
            all_pred_tags.extend(pred_tags)
        else:
            # Este 'else' é uma salvaguarda; o alinhamento deve prevenir isso.
            print(f"AVISO: Desalinhamento de tokens na sentença {i}. Ignorando.")
            print(f"  - Gold: {len(gold_tags)} tags, Pred: {len(pred_tags)} tags")


    # Mapear labels do modelo para o padrão WikiNER (PER, LOC, ORG, MISC)
    # Exemplo: 'B-PERSON' -> 'B-PER'
    label_mapping = {
        'PESSOA': 'PER', 'PERSON': 'PER',
        'LOCAL': 'LOC', 'LOCATION': 'LOC', 'GPE': 'LOC',
        'ORGANIZACAO': 'ORG', 'ORGANIZATION': 'ORG',
        'TEMPO': 'MISC', 'OBRA': 'MISC', 'EVENTO': 'MISC', 'PRODUTO': 'MISC',
        'VALOR': 'MISC', 'LEI': 'MISC', 'JURISPRUDENCIA': 'MISC'
    }

    def map_tag(tag):
        if tag == "O":
            return "O"
        prefix, label = tag.split('-', 1)
        mapped_label = label_mapping.get(label, 'MISC')
        return f"{prefix}-{mapped_label}"

    mapped_pred_tags = [map_tag(tag) for tag in all_pred_tags]

    # Calcular métricas
    # Extrai todas as labels únicas para o classification_report
    labels = sorted(list(set(all_true_tags + mapped_pred_tags) - {'O'}))
    
    report = classification_report(
        all_true_tags,
        mapped_pred_tags,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    return report

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    print("=== ANALISADOR NER - HUGGING FACE (CORRIGIDO) ===")
    
    # --- Configuração ---
    output_dir = "analise_ner_hugging_corrigido"
    os.makedirs(output_dir, exist_ok=True)
    
    # AJUSTE ESTE CAMINHO para o seu arquivo WikiNER
    wikiner_path = "/home/andre/Dev-Ubuntu/IC/Corupus/5462500/aij-wikiner-pt-wp3"
    
    try:
        # --- Carregar Dados ---
        print(f"\nCarregando sentenças WikiNER de: {wikiner_path}")
        sentences = read_wikiner(wikiner_path, max_sentences=1000) # Usar um subconjunto para teste rápido
        print(f"✓ Carregadas {len(sentences)} sentenças")
        
        # --- Carregar Modelo ---
        print("\nInicializando pipeline NER...")
        model_name = "pierreguillou/bert-base-cased-pt-lenerbr"
        device = 0 if torch.cuda.is_available() else -1
        ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            device=device,
            aggregation_strategy="simple" # 'simple' agrupa sub-tokens, o que é ótimo!
        )
        print(f"✓ Pipeline '{model_name}' carregado em {'GPU' if device == 0 else 'CPU'}.")

        # --- Avaliar Modelo ---
        print(f"\nIniciando avaliação NER...")
        report = evaluate_ner_model(
            ner_pipeline, 
            sentences, 
            max_sent=100 # Aumente para uma avaliação mais robusta
        )
        
        # --- Exibir Resultados ---
        print("\n" + "="*60)
        print("RESULTADOS FINAIS - NER HUGGING FACE")
        print("="*60)
        
        # Formata o relatório para exibição com 'tabulate'
        headers = ["Métrica", "Precisão", "Recall", "F1-Score", "Suporte"]
        table_data = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                table_data.append([
                    label,
                    f"{metrics['precision']:.2%}",
                    f"{metrics['recall']:.2%}",
                    f"{metrics['f1-score']:.2%}",
                    int(metrics['support'])
                ])

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\n✓ Análise concluída!")

    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo WikiNER não encontrado em '{wikiner_path}'")
        print("   Por favor, ajuste a variável 'wikiner_path' no script.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a execução: {e}")
        import traceback
        traceback.print_exc()
