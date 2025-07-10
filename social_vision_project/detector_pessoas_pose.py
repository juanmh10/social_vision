
import cv2
import json
from ultralytics import YOLO
import numpy as np

# Carrega o modelo YOLOv8 Pose pré-treinado
model_path = 'yolov8n-pose.pt'
model = YOLO(model_path)

def detectar_pessoas_e_poses(image_path):
    """
    Detecta pessoas e suas poses em uma imagem usando YOLOv8-pose.

    Args:
        image_path (str): O caminho para a imagem de entrada.

    Returns:
        list: Uma lista de dicionários, onde cada dicionário contém
              informações sobre uma pessoa detectada (ID, bbox, keypoints).
    """
    # Lê a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível ler a imagem em {image_path}")
        return []

    # Executa a inferência do modelo na imagem
    results = model(img, verbose=False)

    pessoas_detectadas = []
    # Itera sobre os resultados da detecção
    for i, r in enumerate(results):
        if r.boxes and r.keypoints:
            # Extrai as bounding boxes e os keypoints
            boxes = r.boxes.xyxyn.cpu().numpy()  
            keypoints = r.keypoints.xyn.cpu().numpy()  

            for person_idx in range(len(boxes)): 
                h, w, _ = img.shape
                bbox_abs = [
                    int(boxes[person_idx][0] * w),
                    int(boxes[person_idx][1] * h),
                    int(boxes[person_idx][2] * w),
                    int(boxes[person_idx][3] * h)
                ]
                
                kpts_abs = []
                for kp_idx in range(keypoints[person_idx].shape[0]):
                    kpts_abs.append({
                        "point_id": kp_idx,
                        "x": int(keypoints[person_idx][kp_idx][0] * w),
                        "y": int(keypoints[person_idx][kp_idx][1] * h)
                    })

                pessoa_info = {
                    "id": person_idx,
                    "bbox": bbox_abs, # [x1, y1, x2, y2]
                    "keypoints": kpts_abs
                }
                pessoas_detectadas.append(pessoa_info)

    return pessoas_detectadas

def salvar_resultado_json(data, output_path):
    """Salva os dados em um arquivo JSON."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    caminho_imagem_teste = 'C:/Users/juanm/Desktop/G_CLI-1.0/social_vision_project/sample_image.jpg' # Crie ou adicione uma imagem aqui
    
    try:
        img_teste = cv2.imread(caminho_imagem_teste)
        if img_teste is None:
            print("Criando imagem de placeholder para teste.")
            placeholder_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Adicione uma imagem com pessoas aqui", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.imwrite(caminho_imagem_teste, placeholder_img)
    except Exception as e:
        print(f"Erro ao manusear a imagem de teste: {e}")


    print(f"Executando detecção de pose em: {caminho_imagem_teste}")
    deteccoes = detectar_pessoas_e_poses(caminho_imagem_teste)
    
    if deteccoes:
        caminho_saida_json = 'poses_detectadas.json'
        salvar_resultado_json(deteccoes, caminho_saida_json)
        print(f"Resultados da detecção de pose salvos em: {caminho_saida_json}")
        for p in deteccoes:
            print(f"  - Pessoa ID: {p['id']}, BBox: {p['bbox']}")
    else:
        print("Nenhuma pessoa foi detectada na imagem.")

