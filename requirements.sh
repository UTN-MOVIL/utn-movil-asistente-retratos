#!/bin/bash

# List of packages to install
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
)

# Install each package, continue on failure
for package in "${packages[@]}"; do
    echo "Installing $package..."
    pip install "$package" || echo "Failed to install $package, skipping..."
done

echo "Installation complete!"