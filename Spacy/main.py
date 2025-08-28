import os
import spacy
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from utils.file_readers import parse_conllu, read_wikiner, get_gold_entities
from utils.spacy_processor import load_spacy_model, process_conllu_sentence, process_wikiner_sentence
from metrics.conllu_metrics import calculate_conllu_metrics, analyze_conllu_errors
from metrics.ner_metrics import calculate_ner_metrics
from reports.text_reports import save_conllu_results_to_file, save_conllu_error_analysis, save_ner_results_to_file
from reports.visualizations import visualize_dependencies

def run_conllu_analysis():
    print("\n=== Iniciando Análise CONLL-U ===")
    # 1. Parsear o arquivo CONLL-U
    sentences = parse_conllu(config.CONLLU_FILE)
    print(f"Total de {len(sentences)} sentenças lidas do arquivo CONLL-U.")

    # 2. Carregar o modelo spaCy
    nlp = load_spacy_model(config.SPACY_MODEL)
    print(f"Modelo spaCy \'{config.SPACY_MODEL}\' carregado.")

    # 3. Processar cada sentença com o modelo spaCy (com paralelismo)
    print("Processando sentenças CONLL-U com spaCy (paralelo)... ")
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_conllu_sentence, sent, nlp) for sent in sentences]
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(sentences)} sentenças processadas.")
    print("Processamento CONLL-U concluído.")

    # 4. Calcular as métricas comparativas
    metrics = calculate_conllu_metrics(results)
    print("Métricas CONLL-U calculadas.")

    # 5. Salvar os resultados detalhados e as métricas
    save_conllu_results_to_file(results, metrics, config.CONLLU_RESULTS_FILE)
    print(f"Resultados completos CONLL-U salvos em \'{config.CONLLU_RESULTS_FILE}\'")

    # 6. Gerar análise de erros
    error_analysis = analyze_conllu_errors(results)
    save_conllu_error_analysis(error_analysis, config.CONLLU_ERRORS_FILE)
    print(f"Análise de erros CONLL-U salva em \'{config.CONLLU_ERRORS_FILE}\'")

    # 7. Gerar visualizações de dependências (sem paralelismo para evitar problemas com renderização)
    print("Gerando visualizações de dependências (pode demorar para muitas sentenças)... ")
    visualize_dependencies(sentences, nlp, config.VISUALIZATIONS_DIR)
    print(f"Visualizações de dependências salvas em \'{config.VISUALIZATIONS_DIR}\'")
    print("=== Análise CONLL-U Concluída ===\n")

def run_ner_analysis():
    print("\n=== Iniciando Análise NER ===")
    # 1. Ler o arquivo WikiNER
    sentences_wikiner = read_wikiner(config.WIKINER_FILE, max_sentences=config.MAX_SENTENCES_WIKINER)
    print(f"Total de {len(sentences_wikiner)} sentenças lidas do arquivo WikiNER.")

    # 2. Carregar o modelo spaCy
    nlp = load_spacy_model(config.SPACY_MODEL)
    print(f"Modelo spaCy \'{config.SPACY_MODEL}\' carregado.")

    # 3. Processar cada sentença com o modelo spaCy (com paralelismo)
    print("Processando sentenças WikiNER com spaCy (paralelo)... ")
    gold_entities_list = []
    pred_entities_list = []
    report_data = []

    # Limitar o número de sentenças para o relatório detalhado, se configurado
    sentences_to_process = sentences_wikiner[:config.MAX_SENTENCES_EVALUATE_MODEL]

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, (tokens, tags) in enumerate(sentences_to_process):
            gold_ents = get_gold_entities(tokens, tags)
            gold_entities_list.append(gold_ents)
            futures.append(executor.submit(process_wikiner_sentence, tokens, nlp))

        for i, future in enumerate(as_completed(futures)):
            text, pred_ents = future.result()
            pred_entities_list.append(pred_ents)

            # --- Relatório por sentença ---
            report_data.append(f"=== Sentença {i+1} ===")
            report_data.append(f"Texto: {text}\n")

            gold_text = [f"{ent[0]} ({ent[1]})" for ent in gold_entities_list[i]]
            pred_text = [f"{ent[0]} ({ent[1]})" for ent in pred_entities_list[i]]
            report_data.append("Gold Entities vs. Pred Entities:")
            report_data.append(f"Gold: {gold_text}")
            report_data.append(f"Pred: {pred_text}\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(sentences_to_process)} sentenças processadas para NER.")
    print("Processamento NER concluído.")

    # 4. Calcular as métricas NER
    metrics = calculate_ner_metrics(gold_entities_list, pred_entities_list)
    print("Métricas NER calculadas.")

    # 5. Salvar os resultados NER
    save_ner_results_to_file(metrics, report_data, config.NER_RESULTS_FILE)
    print(f"Resultados NER salvos em \'{config.NER_RESULTS_FILE}\'")
    print("=== Análise NER Concluída ===\n")

if __name__ == "__main__":
    # Criar diretórios de saída se não existirem
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CONLLU_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.NER_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)

    run_conllu_analysis()
    run_ner_analysis()
    print("Todos os processos de análise de NLP foram concluídos!")


