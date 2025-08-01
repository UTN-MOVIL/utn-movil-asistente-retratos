#!/bin/bash

# Asegúrate de que python3.11 esté disponible en tu PATH
if ! command -v python3.11 &> /dev/null
then
    echo "python3.11 no se encontró, por favor instálalo."
    exit 1
fi

# Lista de paquetes a instalar
packages=(
    "roboflow"
    "centernet2"
    "git+https://github.com/openai/CLIP.git"
    "XlsxWriter"
    "kaggle"
    "glasses-detector>=1.0.3"
    "google-api-python-client"
    "google-auth-oauthlib"
    "pandas"
    "openpyxl"
    "autodistill-detic"
    "supervision"
    "scikit-learn"
    "mediapipe"
)

# Instalar cada paquete usando python3.11
for package in "${packages[@]}"; do
    echo "Instalando $package para Python 3.11..."
    python3.11 -m pip install "$package" || echo "Falló la instalación de $package, continuando..."
done

echo "¡Instalación completa!"