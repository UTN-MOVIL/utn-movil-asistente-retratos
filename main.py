import os
from typing import List, Tuple
from deteccion_lentes_v1 import get_glasses_probability
from exportacion_datos_excel import format_to_hyperlinks, normalize_dict_lengths, dict_to_excel, get_file_count

# Import your previously defined function
# from your_module import get_glasses_probability

def process_folder(folder_path: str) -> Tuple[List[str], List[float]]:
    """
    Walks through all files in `folder_path`, collects their paths,
    and computes the glasses probability for each image.

    Returns:
        - A list of file paths
        - A list of corresponding glasses probabilities
    """
    image_paths: List[str] = []
    glasses_probs: List[float] = []

    # Walk through directory (including subdirectories)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                prob = get_glasses_probability(file_path)
            except Exception as e:
                # If inference fails (e.g. non-image file), skip it
                print(f"Skipping {file_path!r}: {e}")
                continue

            image_paths.append(file_path)
            glasses_probs.append(prob)

    return image_paths, glasses_probs


if __name__ == "__main__":
    dataset_folder = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\test"
    results_folder = "results"
    paths, probs = process_folder(dataset_folder)

    informacion = {
        "Rutas": format_to_hyperlinks(paths),
        "Probabilidad de tener lentes": probs
    }

    normalized = normalize_dict_lengths(informacion)
    output_file = dict_to_excel(normalized, f"{results_folder}/Reporte_probabilidad_lentes_{get_file_count(results_folder)+1}.xlsx")
    print(f"Excel file saved to {output_file}")