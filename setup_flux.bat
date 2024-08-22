@echo off
setlocal

:: Step 1: Clone the repository
echo Cloning FLUX repository...
if exist flux (
    echo Directory 'flux' already exists. Deleting...
    rmdir /s /q flux
)
git clone https://github.com/black-forest-labs/flux || (
    echo Error cloning repository. Exiting...
    exit /b 1
)

:: Step 2: Set up Python virtual environment
echo Setting up Python virtual environment...
python -m venv flux\.venv || (
    echo Error setting up virtual environment. Exiting...
    exit /b 1
)
call flux\.venv\Scripts\activate

:: Step 3: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -e "flux/.[all]" || (
    echo Error installing dependencies. Exiting...
    exit /b 1
)

:: Step 4: Create weights directory if not exists
if not exist flux\weights (
    mkdir flux\weights
)

:: Step 5: Download model weights
echo Downloading model weights...
if not exist flux\weights\flux_schnell.safetensors (
    echo "FLUX Schnell model not found. Downloading..."
    curl -L https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/schnell.safetensors -o flux\weights\flux_schnell.safetensors
)
if not exist flux\weights\flux_dev.safetensors (
    echo "FLUX Dev model not found. Downloading..."
    curl -L https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/dev.safetensors -o flux\weights\flux_dev.safetensors
)
if not exist flux\weights\autoencoder.safetensors (
    echo "Autoencoder model not found. Downloading..."
    curl -L https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/autoencoder.safetensors -o flux\weights\autoencoder.safetensors
)

:: Step 6: Export environment variables
echo Setting environment variables...
set FLUX_SCHNELL=%CD%\flux\weights\flux_schnell.safetensors
set FLUX_DEV=%CD%\flux\weights\flux_dev.safetensors
set AE=%CD%\flux\weights\autoencoder.safetensors

:: Step 7: Choose demo to run
echo.
echo Choose a demo to run:
echo 1. Gradio
echo 2. Streamlit
echo 3. CLI
set /p DEMO_CHOICE="Enter the number of your choice: "

if "%DEMO_CHOICE%"=="1" (
    echo Running Gradio demo...
    python flux\demo_gr.py --name flux-schnell --device cuda || (
        echo Error running Gradio demo. Exiting...
        exit /b 1
    )
) else if "%DEMO_CHOICE%"=="2" (
    echo Running Streamlit demo...
    streamlit run flux\demo_st.py || (
        echo Error running Streamlit demo. Exiting...
        exit /b 1
    )
) else if "%DEMO_CHOICE%"=="3" (
    echo Running CLI demo...
    python -m flux --name flux-schnell --loop || (
        echo Error running CLI demo. Exiting...
        exit /b 1
    )
) else (
    echo Invalid choice. Exiting...
    exit /b 1
)

echo Setup complete!
endlocal
