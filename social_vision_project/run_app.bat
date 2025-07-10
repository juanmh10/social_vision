@echo off
REM Este script inicia o aplicativo de analise de interacao social.

echo Iniciando o servidor Streamlit...
echo Navegando para o diretorio do projeto...

REM Navega para o diretorio onde o script esta localizado.
cd /d "%~dp0"

echo Lancando o aplicativo... Uma janela do navegador deve abrir em breve.

REM Executa o aplicativo Streamlit usando o interpretador Python do ambiente virtual.
"%~dp0\venv\Scripts\python.exe" -m streamlit run app_streamlit.py

echo O servidor foi encerrado. Pressione qualquer tecla para fechar esta janela.
pause