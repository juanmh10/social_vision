
import cv2
import mediapipe as mp
import json
import numpy as np

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def analisar_gesticulacao(deteccoes_pessoas):
    """
    Analisa se há gesticulação ativa com base na posição das mãos em relação ao corpo.

    Args:
        deteccoes_pessoas (list): Lista de detecções de pessoas com keypoints.

    Returns:
        list: A lista de detecções atualizada com a feature de gesticulação.
    """
    for pessoa in deteccoes_pessoas:
        if 'keypoints' not in pessoa or not pessoa['keypoints']:
            pessoa['gesticulando'] = False
            continue

        keypoints = {kp['point_id']: kp for kp in pessoa['keypoints']}

        # Índices dos keypoints do MediaPipe Pose (via YOLOv8-Pose)
        # Ombros
        ombro_esq_id, ombro_dir_id = 5, 6
        # Pulsos
        pulso_esq_id, pulso_dir_id = 9, 10
        # Quadris
        quadril_esq_id, quadril_dir_id = 11, 12

        gesticulando = False
        try:
            # Pega as coordenadas Y dos ombros, pulsos e quadris
            y_ombro_esq = keypoints[ombro_esq_id]['y']
            y_ombro_dir = keypoints[ombro_dir_id]['y']
            y_pulso_esq = keypoints[pulso_esq_id]['y']
            y_pulso_dir = keypoints[pulso_dir_id]['y']
            y_quadril_esq = keypoints[quadril_esq_id]['y']
            y_quadril_dir = keypoints[quadril_dir_id]['y']

            # Calcula a linha média do tronco (entre ombros e quadris)
            linha_media_ombros = (y_ombro_esq + y_ombro_dir) / 2
            linha_media_quadris = (y_quadril_esq + y_quadril_dir) / 2

            # Heurística: se qualquer um dos pulsos estiver acima da linha dos ombros
            # ou significativamente acima da linha do quadril, consideramos gesticulação.
            # Isso indica que as mãos estão levantadas, e não em repouso.
            if (y_pulso_esq < linha_media_ombros or y_pulso_dir < linha_media_ombros or 
                y_pulso_esq < (linha_media_quadris - (linha_media_quadris - linha_media_ombros) * 0.2)):
                gesticulando = True

        except KeyError:
            # Caso algum keypoint essencial não seja detectado
            gesticulando = False

        pessoa['gesticulando'] = gesticulando

    return deteccoes_pessoas

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
    # Este script usa os keypoints do primeiro detector
    caminho_poses_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/poses_detectadas.json'
    caminho_expressoes_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/expressoes_faciais.json'

    # Carregamos os dados de expressões, que já contêm os dados de pose
    dados_atuais = carregar_dados_json(caminho_expressoes_json)

    if dados_atuais:
        print("Analisando gestos...")
        resultado_gestos = analisar_gesticulacao(dados_atuais)
        
        caminho_saida_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/gestos.json'
        salvar_resultado_json(resultado_gestos, caminho_saida_json)
        print(f"Resultados da análise de gestos salvos em: {caminho_saida_json}")

        for p in resultado_gestos:
            print(f"  - Pessoa ID: {p['id']}, Gesticulando: {p.get('gesticulando', 'N/A')}")
    else:
        print("Nenhum dado de entrada encontrado. Execute os scripts anteriores.")
