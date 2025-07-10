
import json
import numpy as np

def estimar_vetor_olhar(keypoints_pessoa):
    """
    Estima um vetor de direção do olhar (simplificado).
    Usa a posição do nariz e a média dos olhos para criar um vetor 2D.
    
    Args:
        keypoints_pessoa (dict): Dicionário de keypoints da pessoa.

    Returns:
        np.array: O vetor de direção do olhar (ou None se não for possível calcular).
    """
    try:
        # Índices dos keypoints (YOLOv8-Pose)
        nariz_id = 0
        olho_esq_id, olho_dir_id = 1, 2

        ponto_nariz = np.array([keypoints_pessoa[nariz_id]['x'], keypoints_pessoa[nariz_id]['y']])
        ponto_olho_esq = np.array([keypoints_pessoa[olho_esq_id]['x'], keypoints_pessoa[olho_esq_id]['y']])
        ponto_olho_dir = np.array([keypoints_pessoa[olho_dir_id]['x'], keypoints_pessoa[olho_dir_id]['y']])
  
        centro_olhos = (ponto_olho_esq + ponto_olho_dir) / 2
        
        # O vetor do centro dos olhos para o nariz indica a direção do rosto
        vetor_direcao = ponto_nariz - centro_olhos
        
        # Normaliza o vetor
        norma = np.linalg.norm(vetor_direcao)
        if norma == 0:
            return None
        
        return vetor_direcao / norma
    except (KeyError, IndexError):
        return None

def analisar_direcao_olhar(deteccoes):
    """
    Analisa para onde cada pessoa está olhando.

    Args:
        deteccoes (list): Lista de detecções de pessoas.

    Returns:
        list: A lista de detecções atualizada com a informação do alvo do olhar.
    """
    # Primeiro, calcula o centro de cada pessoa (usando a bbox)
    for pessoa in deteccoes:
        if 'bbox' in pessoa:
            x1, y1, x2, y2 = pessoa['bbox']
            pessoa['centro'] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        else:
            pessoa['centro'] = None

    # Agora, analisa o olhar de cada pessoa
    for i, pessoa_olhando in enumerate(deteccoes):
        pessoa_olhando['olhando_para_id'] = None
        
        if 'keypoints' not in pessoa_olhando or not pessoa_olhando['keypoints'] or pessoa_olhando['centro'] is None:
            continue

        keypoints_dict = {kp['point_id']: kp for kp in pessoa_olhando['keypoints']}
        vetor_olhar = estimar_vetor_olhar(keypoints_dict)

        if vetor_olhar is None:
            continue

        melhor_alvo_id = -1
        menor_angulo = np.pi # Inicia com 180 graus

        # Verifica para quem a pessoa pode estar olhando
        for j, pessoa_alvo in enumerate(deteccoes):
            if i == j or pessoa_alvo['centro'] is None: # Não pode olhar para si mesma
                continue

            # Vetor da pessoa que olha para o alvo
            vetor_para_alvo = pessoa_alvo['centro'] - pessoa_olhando['centro']
            norma = np.linalg.norm(vetor_para_alvo)
            if norma == 0:
                continue
            vetor_para_alvo_norm = vetor_para_alvo / norma

            cos_theta = np.dot(vetor_olhar, vetor_para_alvo_norm)
            angulo = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # Clip para evitar erros de precisão

            if angulo < np.pi / 6 and angulo < menor_angulo:
                menor_angulo = angulo
                melhor_alvo_id = pessoa_alvo['id']
        
        if melhor_alvo_id != -1:
            pessoa_olhando['olhando_para_id'] = melhor_alvo_id

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
    caminho_gestos_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/gestos.json'

    dados_atuais = carregar_dados_json(caminho_gestos_json)

    if dados_atuais:
        print("Analisando direção do olhar...")
        resultado_olhar = analisar_direcao_olhar(dados_atuais)
        
        caminho_saida_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/direcao_olhar.json'
        salvar_resultado_json(resultado_olhar, caminho_saida_json)
        print(f"Resultados da análise de olhar salvos em: {caminho_saida_json}")

        for p in resultado_olhar:
            alvo_id = p.get('olhando_para_id')
            if alvo_id is not None:
                print(f"  - Pessoa ID: {p['id']} está olhando para a Pessoa ID: {alvo_id}")
            else:
                print(f"  - Pessoa ID: {p['id']} não parece estar olhando para ninguém.")
    else:
        print("Nenhum dado de entrada encontrado. Execute os scripts anteriores.")
