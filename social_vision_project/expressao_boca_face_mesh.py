
import cv2
import mediapipe as mp
import json
import numpy as np

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def calcular_distancia_vertical(ponto1, ponto2):
    """Calcula a distância euclidiana vertical entre dois pontos."""
    return abs(ponto1.y - ponto2.y)

def analisar_expressoes_faciais(image_path, deteccoes_com_faces):
    """
    Analisa expressões faciais, como a abertura da boca, usando MediaPipe Face Mesh.

    Args:
        image_path (str): Caminho para a imagem original.
        deteccoes_com_faces (list): Lista de detecções com informações de face.

    Returns:
        list: A lista de detecções atualizada com features de expressão.
    """
    img = cv2.imread(image_path)
    if img is None:
        return deteccoes_com_faces

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pessoa in deteccoes_com_faces:
        if not pessoa.get('face_info') or not pessoa['face_info'].get('face_bbox'):
            pessoa['expressoes'] = {"boca_aberta": False, "olhos_fechados": False}
            continue

        # Recorta a região da face
        x1, y1, x2, y2 = pessoa['face_info']['face_bbox']
        roi_face = img_rgb[y1:y2, x1:x2]

        if roi_face.size == 0:
            pessoa['expressoes'] = {"boca_aberta": False, "olhos_fechados": False}
            continue

        # Processa a ROI da face com o Face Mesh
        results = face_mesh.process(roi_face)

        boca_aberta = False
        olhos_fechados = False 

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ponto_labio_superior = face_landmarks.landmark[13]
                ponto_labio_inferior = face_landmarks.landmark[14]
                ponto_canto_boca_esq = face_landmarks.landmark[61]
                ponto_canto_boca_dir = face_landmarks.landmark[291]

                # Calcula a abertura vertical e horizontal da boca
                abertura_vertical = calcular_distancia_vertical(ponto_labio_superior, ponto_labio_inferior)
                abertura_horizontal = calcular_distancia_vertical(ponto_canto_boca_esq, ponto_canto_boca_dir)
                
                if abertura_horizontal > 0 and (abertura_vertical / abertura_horizontal) > 0.4:
                    boca_aberta = True

                # TODO: Adicionar lógica para olhos fechados se necessário            

        pessoa['expressoes'] = {
            "boca_aberta": boca_aberta,
            "olhos_fechados": olhos_fechados
        }

    return deteccoes_com_faces

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
    caminho_imagem = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/sample_image.jpg'
    caminho_faces_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/faces_detectadas.json'

    deteccoes_face = carregar_dados_json(caminho_faces_json)

    if deteccoes_face:
        print("Analisando expressões faciais...")
        resultado_expressoes = analisar_expressoes_faciais(caminho_imagem, deteccoes_face)
        
        caminho_saida_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/expressoes_faciais.json'
        salvar_resultado_json(resultado_expressoes, caminho_saida_json)
        print(f"Resultados da análise de expressão salvos em: {caminho_saida_json}")

        for p in resultado_expressoes:
            if 'expressoes' in p:
                print(f"  - Pessoa ID: {p['id']}, Boca Aberta: {p['expressoes']['boca_aberta']}")
    else:
        print("Nenhum dado de faces encontrado. Execute 'detector_faces.py' primeiro.")
