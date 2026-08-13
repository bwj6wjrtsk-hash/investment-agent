@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run research_system\ui.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
