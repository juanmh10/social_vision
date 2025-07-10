
import cv2
import face_recognition
import json
import numpy as np

def detectar_faces(image_path, deteccoes_pessoas):
    """
    Detecta faces nas áreas das pessoas detectadas.

    Args:
        image_path (str): Caminho para a imagem original.
        deteccoes_pessoas (list): Lista de dicionários com as detecções de pessoas.

    Returns:
        list: A lista de detecções de pessoas atualizada com informações das faces.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao ler a imagem: {image_path}")
        return deteccoes_pessoas

    # Converte a imagem de BGR (OpenCV) para RGB (face_recognition)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pessoa in deteccoes_pessoas:
        # Extrai a bounding box da pessoa
        x1, y1, x2, y2 = pessoa['bbox']
        # Garante que as coordenadas estão dentro dos limites da imagem
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        roi_pessoa = img_rgb[y1:y2, x1:x2].copy()

        if roi_pessoa.size == 0:
            pessoa['face_info'] = None
            continue
            
        print(f"Debug (Pessoa ID {pessoa['id']}): ROI shape: {roi_pessoa.shape}, ROI dtype: {roi_pessoa.dtype}")

        face_locations = face_recognition.face_locations(roi_pessoa, model="hog") 
        
        # --- DEBUG ---
        print(f"Debug (Pessoa ID {pessoa['id']}): Encontradas {len(face_locations)} faces com o modelo 'hog'.")

        if not face_locations:
            pessoa['face_info'] = None
            continue

        face_encodings = face_recognition.face_encodings(roi_pessoa, face_locations)

        if face_encodings:
            # Pega a maior face encontrada na ROI, caso haja mais de uma
            main_face_loc = max(face_locations, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
            main_face_encoding = face_encodings[face_locations.index(main_face_loc)]

            top, right, bottom, left = main_face_loc
            face_bbox_abs = [left + x1, top + y1, right + x1, bottom + y1]
            
            pessoa['face_info'] = {
                "face_bbox": face_bbox_abs,
                "face_encoding": main_face_encoding.tolist() # Converte para lista para ser serializável em JSON
            }
        else:
            pessoa['face_info'] = None # Nenhuma face encontrada para esta pessoa

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
    caminho_imagem = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/sample_image.jpg'
    caminho_poses_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/poses_detectadas.json'
    
    # Carrega as detecções de pose do script anterior
    deteccoes_pose = carregar_dados_json(caminho_poses_json)

    if deteccoes_pose:
        print("Executando detecção de faces...")
        # Executa a detecção de faces
        resultado_com_faces = detectar_faces(caminho_imagem, deteccoes_pose)
        
        caminho_saida_json = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/faces_detectadas.json'
        salvar_resultado_json(resultado_com_faces, caminho_saida_json)
        print(f"Resultados da detecção de faces salvos em: {caminho_saida_json}")

        for p in resultado_com_faces:
            if p['face_info']:
                print(f"  - Pessoa ID: {p['id']}, Face BBox: {p['face_info']['face_bbox']}")
            else:
                print(f"  - Pessoa ID: {p['id']}, Nenhuma face detectada.")
    else:
        print("Nenhum dado de pose encontrado. Execute 'detector_pessoas_pose.py' primeiro.")
