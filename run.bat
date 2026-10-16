@echo off

cd /d "C:\Users\ahmad\OneDrive\Desktop\Projects"

REM Activate the virtual environment (CMD/Batch version, not PowerShell)
call .venv\Scripts\activate.bat

REM Go into the folder that has app.py
cd "SatelliteGraph"

REM Run the Streamlit app
streamlit run ".\thesis_dashboard.py"
