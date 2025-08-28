import os

# Caminhos dos arquivos de entrada
CONLLU_FILE = os.path.expanduser("/home/anrdre/IC-NLP/UD_Portuguese-Bosque-master/pt_bosque-ud-test.conllu")
WIKINER_FILE = os.path.expanduser("/home/anrdre/IC-NLP/5462500/aij-wikiner-pt-wp3")

# Modelo spaCy a ser utilizado
SPACY_MODEL = "pt_core_news_lg"

# Diretórios de saída
OUTPUT_DIR = "./output"
CONLLU_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "conllu_analysis")
NER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ner_analysis")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Criar diretórios de saída se não existirem
os.makedirs(CONLLU_OUTPUT_DIR, exist_ok=True)
os.makedirs(NER_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Nomes dos arquivos de saída
CONLLU_RESULTS_FILE = os.path.join(CONLLU_OUTPUT_DIR, "resultados_completos.txt")
CONLLU_ERRORS_FILE = os.path.join(CONLLU_OUTPUT_DIR, "analise_erros.txt")
NER_RESULTS_FILE = os.path.join(NER_OUTPUT_DIR, "resultado_NER.txt")

# Parâmetros para processamento
MAX_SENTENCES_WIKINER = 10000
MAX_SENTENCES_EVALUATE_MODEL = 500


