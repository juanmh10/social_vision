
import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np


from detector_pessoas_pose import detectar_pessoas_e_poses
from detector_faces import detectar_faces
from expressao_boca_face_mesh import analisar_expressoes_faciais
from analise_pose_gestos import analisar_gesticulacao
from direcao_olhar import analisar_direcao_olhar
from classificador_social import classificar_papeis_sociais
from app_teste import desenhar_resultados 

st.set_page_config(
    page_title="Analisador de Interação Social",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções Auxiliares ---
def run_pipeline(image_path):
    """Executa o pipeline completo e retorna os resultados e a imagem final."""
    # Etapa 1: Detecção de Pessoas e Pose
    deteccoes_pose = detectar_pessoas_e_poses(image_path)
    if not deteccoes_pose:
        return None, None # Retorna None se ninguém for detectado

    # Etapas subsequentes
    deteccoes_face = detectar_faces(image_path, deteccoes_pose)
    deteccoes_expressoes = analisar_expressoes_faciais(image_path, deteccoes_face)
    deteccoes_gestos = analisar_gesticulacao(deteccoes_expressoes)
    deteccoes_olhar = analisar_direcao_olhar(deteccoes_gestos)
    resultado_final = classificar_papeis_sociais(deteccoes_olhar)

    # Desenha o resultado na imagem
    img = cv2.imread(image_path)
    img_resultado = desenhar_resultados(img, resultado_final)
    
    return resultado_final, img_resultado

# --- Interface Principal ---
st.title("👥 Analisador de Interação Social em Imagens")
st.markdown("""
Esta aplicação utiliza um pipeline de Visão Computacional para detectar pessoas em uma imagem e classificá-las como **Falando** ou **Ouvindo**.

O modelo analisa múltiplos fatores:
- **Pose Corporal e Gestos:** Para identificar gesticulação ativa.
- **Expressões Faciais:** Foco na abertura da boca como principal indicador de fala.
- **Direção do Olhar:** Para inferir para quem uma pessoa está prestando atenção.

**Instruções:**
1.  Faça o upload de uma imagem no painel à esquerda.
2.  Aguarde a análise ser concluída.
3.  Veja o resultado na tela!
""")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("⚙️ Configurações")
uploaded_file = st.sidebar.file_uploader(
    "Escolha uma imagem (JPG, PNG)", 
    type=['jpg', 'jpeg', 'png']
)

st.sidebar.markdown("--- ")
st.sidebar.info("Projeto desenvolvido para demonstrar um pipeline de análise de comportamento social.")

# --- Lógica Principal da Aplicação ---
if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    temp_dir = "."
    temp_image_path = os.path.join(temp_dir, "temp_uploaded_image.jpg")
    cv2.imwrite(temp_image_path, opencv_image)

    # Exibe a imagem original
    st.header("🖼️ Imagem Original")
    st.image(opencv_image, channels="BGR", caption="Imagem enviada pelo usuário.", use_column_width=True)

    st.markdown("--- ")

    # Adiciona um botão para iniciar a análise
    if st.button("🚀 Iniciar Análise Agora", use_container_width=True):
        # Executa o pipeline
        with st.spinner('🧠 Analisando a imagem... Isso pode levar alguns segundos... '):
            resultados_json, imagem_processada = run_pipeline(temp_image_path)

        if resultados_json is None:
            st.warning("⚠️ Nenhuma pessoa foi detectada na imagem. Tente outra imagem.")
        else:
            st.success("✅ Análise concluída com sucesso!")
            st.balloons()

            # --- Layout de Duas Colunas para o Resultado ---
            col1, col2 = st.columns([2, 1]) 

            with col1:
                st.header("📊 Resultado da Análise")
                st.image(imagem_processada, channels="BGR", caption="Imagem com os papéis sociais identificados.", use_column_width=True)

            with col2:
                st.header("📖 Legenda e Detalhes")
                
                st.markdown("""
                <style>
                .legend-color-box {
                    width: 20px;
                    height: 20px;
                    display: inline-block;
                    border: 1px solid #ccc;
                    margin-right: 10px;
                    vertical-align: middle;
                }
                .legend-container {
                    margin-bottom: 20px;
                }
                </style>
                
                <div class="legend-container">
                    <div>
                        <span class="legend-color-box" style="background-color: #00FF00;"></span>
                        <strong>Falando</strong>
                    </div>
                    <div style="margin-top: 10px;">
                        <span class="legend-color-box" style="background-color: #FFA500;"></span>
                        <strong>Ouvindo</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Resumo do Workflow
                st.markdown("---")
                st.subheader("Como o modelo funciona?")
                st.markdown("""
                O resultado é gerado por um pipeline de 6 etapas:
                1.  **Detecção de Pessoas e Pose:** Encontra pessoas e seus esqueletos (`YOLOv8`).
                2.  **Detecção de Rosto:** Localiza o rosto de cada pessoa.
                3.  **Análise de Expressão:** Verifica se a boca está aberta (`MediaPipe`).
                4.  **Análise de Gestos:** Avalia se as mãos estão gesticulando.
                5.  **Direção do Olhar:** Estima para quem a pessoa está olhando.
                6.  **Classificação Final:** Uma lógica combina todas as pistas para definir o papel social.
                """)

                # Expander para mostrar os dados JSON
                with st.expander("📄 Ver detalhes técnicos (JSON)"):
                    st.json(resultados_json)

else:
    st.info("Aguardando o upload de uma imagem para iniciar a análise.")

