
import json

def classificar_papeis_sociais(deteccoes):
    """
    Classifica as pessoas como 'Falando' ou 'Ouvindo' com base nas features extraídas.

    Args:
        deteccoes (list): Lista de detecções com todas as features.

    Returns:
        list: A lista de detecções com a classificação final.
    """
    ids_falando = set()
    ids_ouvindo = set()

    # Primeira passagem: identificar quem está claramente falando
    for pessoa in deteccoes:
        # Features da pessoa
        boca_aberta = pessoa.get('expressoes', {}).get('boca_aberta', False)
        gesticulando = pessoa.get('gesticulando', False)

        # Lógica de classificação para "Falando"
        # Ter a boca aberta é o indicador mais forte.
        # Gesticular aumenta a confiança.
        if boca_aberta:
            pessoa['papel_social'] = 'Falando'
            ids_falando.add(pessoa['id'])
        else:
            pessoa['papel_social'] = 'Indeterminado' # Estado inicial

    # Segunda passagem: identificar quem está ouvindo
    for pessoa in deteccoes:
        if pessoa['papel_social'] == 'Indeterminado':
            olhando_para_id = pessoa.get('olhando_para_id')

            # Se a pessoa está olhando para alguém que está falando, ela é uma ouvinte.
            if olhando_para_id is not None and olhando_para_id in ids_falando:
                pessoa['papel_social'] = 'Ouvindo'
                ids_ouvindo.add(pessoa['id'])
            else:               
                pessoa['papel_social'] = 'Ouvindo'
                ids_ouvindo.add(pessoa['id'])

    if not ids_falando:
        for pessoa in deteccoes:
            if pessoa.get('gesticulando', False):
                pessoa['papel_social'] = 'Falando'               
                falante_id = pessoa['id']
                for outro in deteccoes:
                    if outro['id'] != falante_id and outro.get('olhando_para_id') == falante_id:
                        outro['papel_social'] = 'Ouvindo'
                break 

    return deteccoes

def carregar_dados_json(json_path):
    """Carrega dados de um arquivo JSON."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Arquivo JSON não encontrado: {json_path}")
        return []

def salvar_resultado_json(data, output_path):
    """Salva os dados em um arquivo JSON."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    caminho_olhar_json = 'direcao_olhar.json'

    dados_atuais = carregar_dados_json(caminho_olhar_json)

    if dados_atuais:
        print("Classificando papéis sociais...")
        resultado_final = classificar_papeis_sociais(dados_atuais)

        caminho_saida_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/classificacao_final.json'
        salvar_resultado_json(resultado_final, caminho_saida_json)
        print(f"Resultados da classificação salvos em: {caminho_saida_json}")

        for p in resultado_final:
            print(f"  - Pessoa ID: {p['id']}, Papel: {p.get('papel_social', 'N/A')}")
    else:
        print("Nenhum dado de entrada encontrado. Execute os scripts anteriores.")
