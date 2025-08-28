import os
from spacy import displacy

def visualize_dependencies(sentences, nlp_model, output_dir):
    """
    Gera visualizações gráficas das árvores de dependência para cada sentença.
    """
    nlp = nlp_model # nlp_model is already the loaded spacy model
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se ele não existir
    for i, sent in enumerate(sentences):
        text = sent["text"]
        doc = nlp(text)
        svg = displacy.render(doc, style="dep", jupyter=False)
        with open(f"{output_dir}/sentence_{i+1}.svg", "w", encoding="utf-8") as f:
            f.write(svg)
    print(f"Visualizações de dependências salvas em \'{output_dir}\'")


