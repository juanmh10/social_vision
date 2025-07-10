import cv2
import os
import json

# Importa as funções dos outros scripts
from detector_pessoas_pose import detectar_pessoas_e_poses
from detector_faces import detectar_faces
from expressao_boca_face_mesh import analisar_expressoes_faciais
from analise_pose_gestos import analisar_gesticulacao
from direcao_olhar import analisar_direcao_olhar
from classificador_social import classificar_papeis_sociais

def desenhar_resultados(image, resultados):
    """
    Desenha as bounding boxes e os rótulos de classificação na imagem.
    """
    for pessoa in resultados:
        x1, y1, x2, y2 = pessoa['bbox']
        papel = pessoa.get('papel_social', 'Indeterminado')

        # Define a cor com base no papel social
        if papel == 'Falando':
            cor = (0, 255, 0) # Verde (BGR)
        elif papel == 'Ouvindo':
            cor = (0, 165, 255) # Laranja (BGR)
        else:
            cor = (0, 0, 255) # Vermelho (BGR)

        cv2.rectangle(image, (x1, y1), (x2, y2), cor, 3)

    return image

def main(image_path):
    """
    Executa o pipeline completo de análise de papéis sociais.
    """
    if not os.path.exists(image_path):
        print(f"Erro: A imagem de entrada não foi encontrada em {image_path}")
        return

    print("--- Iniciando Pipeline de Análise de Papel Social ---")

    # Etapa 1: Detecção de Pessoas e Pose
    print("\n[ETAPA 1/6] Detectando pessoas e poses...")
    deteccoes_pose = detectar_pessoas_e_poses(image_path)
    if not deteccoes_pose:
        print("Nenhuma pessoa detectada. Encerrando.")
        return
    print(f"{len(deteccoes_pose)} pessoa(s) detectada(s).")

    # Etapa 2: Detecção de Faces
    print("\n[ETAPA 2/6] Detectando faces...")
    deteccoes_face = detectar_faces(image_path, deteccoes_pose)
    print("Detecção de faces concluída.")

    # Etapa 3: Análise de Expressões Faciais (Boca Aberta)
    print("\n[ETAPA 3/6] Analisando expressões faciais...")
    deteccoes_expressoes = analisar_expressoes_faciais(image_path, deteccoes_face)
    print("Análise de expressões concluída.")

    # Etapa 4: Análise de Gestos
    print("\n[ETAPA 4/6] Analisando gestos...")
    deteccoes_gestos = analisar_gesticulacao(deteccoes_expressoes)
    print("Análise de gestos concluída.")

    # Etapa 5: Análise da Direção do Olhar
    print("\n[ETAPA 5/6] Analisando direção do olhar...")
    deteccoes_olhar = analisar_direcao_olhar(deteccoes_gestos)
    print("Análise do olhar concluída.")

    # Etapa 6: Classificação do Papel Social
    print("\n[ETAPA 6/6] Classificando papéis sociais...")
    resultado_final = classificar_papeis_sociais(deteccoes_olhar)
    print("Classificação concluída.")

    # Salva o resultado final em JSON para depuração
    output_dir = os.path.dirname(image_path)
    final_json_path = os.path.join(output_dir, "resultado_completo.json")
    with open(final_json_path, 'w') as f:
        json.dump(resultado_final, f, indent=4)
    print(f"\nResultado final salvo em: {final_json_path}")

    # Exibição do resultado final
    print("\n--- Exibindo resultado final --- ")
    img = cv2.imread(image_path)
    img_resultado = desenhar_resultados(img, resultado_final)

 
    h, w, _ = img_resultado.shape
    max_dim = 1080
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img_resultado = cv2.resize(img_resultado, (int(w * scale), int(h * scale)))

    cv2.imshow('Classificacao de Papeis Sociais', img_resultado)
    print("Pressione qualquer tecla para fechar a janela da imagem.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Caminho para a imagem que você quer analisar
    caminho_da_imagem = 'sample_image.jpg'
    main(caminho_da_imagem)

