# 👥 Analisador de Interação Social em Imagens

Este projeto utiliza um pipeline de Visão Computacional para detectar pessoas em uma imagem e classificar seus papéis sociais primários como **"Falando"** ou **"Ouvindo"**. A análise é baseada em múltiplos fatores, incluindo pose corporal, expressões faciais e direção do olhar.

A interface principal é uma aplicação web criada com Streamlit, que permite ao usuário fazer o upload de uma imagem e visualizar o resultado da análise de forma interativa.

## ✨ Funcionalidades

-   **Detecção de Múltiplas Pessoas:** Usa o modelo YOLOv8-Pose para localizar todas as pessoas na imagem.
-   **Análise de Pistas Visuais:** Extrai informações sobre gestos, abertura da boca e direção do olhar.
-   **Classificação de Papel Social:** Aplica uma lógica heurística para determinar se uma pessoa está falando ou ouvindo.
-   **Interface Web Interativa:** Permite o upload fácil de imagens e a visualização clara dos resultados.
-   **Execução Simplificada:** Um arquivo de lote (`run_app.bat`) permite iniciar a aplicação com um duplo clique no Windows.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

-   Python 3.10+
-   Um ambiente virtual Python (o projeto foi configurado para usar um localizado em `../venv`).

### 1. Instalação das Dependências

Todas as dependências necessárias estão listadas no arquivo `requirements.txt`. Para instalá-las, use o `pip.exe` específico do seu ambiente virtual. Estando na pasta raíz do projeto (`Clas_Talk_Listsen`), execute o seguinte comando no terminal:


### 2. Iniciando a Aplicação (Método Recomendado)

Para facilitar o uso, foi criado um script de execução.

1.  Navegue até a pasta `social_vision_project`.
2.  Dê um duplo clique no arquivo `run_app.bat`.

Isso iniciará o servidor Streamlit e abrirá a aplicação web no seu navegador padrão.

---

## ⚙️ Como Funciona: O Pipeline de Análise

O resultado final é gerado por um pipeline modular que executa 6 etapas em sequência. Cada etapa é implementada em seu próprio script Python.

1.  **`detector_pessoas_pose.py`**
    -   **O que faz:** Carrega a imagem e usa o modelo **YOLOv8-Pose** para detectar todas as pessoas presentes.
    -   **Saída:** Para cada pessoa, extrai a caixa delimitadora (bounding box) e os pontos-chave do esqueleto (keypoints).

2.  **`detector_faces.py`**
    -   **O que faz:** Para cada pessoa detectada, usa a biblioteca **`face_recognition`** (baseada no `dlib`) para localizar a região do rosto.
    -   **Saída:** Adiciona a caixa delimitadora do rosto aos dados da pessoa.

3.  **`expressao_boca_face_mesh.py`**
    -   **O que faz:** Utiliza o **MediaPipe Face Mesh** para mapear uma malha 3D detalhada sobre cada rosto. Analisa os pontos da boca para determinar se ela está aberta, um forte indicador de fala.
    -   **Saída:** Uma flag booleana `boca_aberta`.

4.  **`analise_pose_gestos.py`**
    -   **O que faz:** Analisa os keypoints do esqueleto (obtidos na etapa 1) para verificar se as mãos da pessoa estão levantadas em uma posição de gesticulação ativa.
    -   **Saída:** Uma flag booleana `gesticulando`.

5.  **`direcao_olhar.py`**
    -   **O que faz:** Estima a direção do olhar de cada pessoa com base na posição dos olhos e do nariz. Em seguida, calcula se esse "vetor de atenção" está apontando para outra pessoa na cena.
    -   **Saída:** O ID da pessoa para quem o indivíduo está olhando (`olhando_para_id`).

6.  **`classificador_social.py`**
    -   **O que faz:** Reúne todas as pistas coletadas (`boca_aberta`, `gesticulando`, `olhando_para_id`) e aplica uma lógica de decisão para classificar cada pessoa.
    -   **Lógica:** Uma pessoa com a boca aberta é quase sempre "Falando". Uma pessoa que está olhando para quem fala é classificada como "Ouvindo".
    -   **Saída:** O rótulo final: "Falando" ou "Ouvindo".

---

## 🧠 Sobre os Modelos (Treinamento vs. Inferência)

É fundamental entender que **este projeto não realiza nenhum treinamento de modelos**. Ele exclusivamente utiliza modelos de aprendizado profundo que foram **pré-treinados** por outras equipes (Google, Ultralytics, etc.) em conjuntos de dados massivos. O processo que executamos é chamado de **inferência**.

### Onde os modelos são salvos?

As bibliotecas que usamos gerenciam o download e o armazenamento dos modelos de forma automática. Na primeira vez que o código é executado, os modelos são baixados e salvos em um diretório de cache local no seu computador.

-   **YOLOv8 (`ultralytics`):** Salva os modelos (`.pt`) em `C:\Users\<Seu-Usuario>\AppData\Roaming\Ultralytics\models\`.
-   **MediaPipe (`mediapipe`):** Gerencia seus modelos (`.tflite`) internamente, geralmente dentro da pasta da biblioteca ou em um local de cache próprio.
-   **`face_recognition` (`dlib`):** Salva seus modelos (`.dat`) em um diretório de cache gerenciado pela biblioteca.

Esse mecanismo garante que os modelos sejam baixados apenas uma vez, tornando as execuções futuras muito mais rápidas.

```