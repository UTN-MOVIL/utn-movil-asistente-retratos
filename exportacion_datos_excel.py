import pandas as pd
import os

def dict_to_excel(informacion, filename="informacion.xlsx"):
    """
    Convierte un diccionario en un .xlsx, centrándolo y dibujando
    bordes negros sólo en la tabla de datos.
    """
    df = pd.DataFrame(informacion)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Volcar DataFrame en la hoja
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Formato para datos: centrado + bordes negros
        data_fmt = workbook.add_format({
            'align':       'center',
            'valign':      'vcenter',
            'border':      1,
            'border_color':'black'
        })

        # Formato para header: como el anterior + negrita
        header_fmt = workbook.add_format({
            'align':       'center',
            'valign':      'vcenter',
            'border':      1,
            'border_color':'black',
            'bold':        True
        })

        # Aplicar el formato solo a las celdas con datos en la tabla
        # Para el header (fila 0) con formato en negrita
        for col_idx in range(len(df.columns)):
            worksheet.write(0, col_idx, df.columns[col_idx], header_fmt)
        
        # Para los datos (filas 1 a n) sin negrita
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                # +1 en la fila porque la fila 0 es el header
                worksheet.write(row + 1, col, df.iloc[row, col], data_fmt)

        # Ajustar anchura de columnas
        for col_idx, col in enumerate(df.columns):
            worksheet.set_column(col_idx, col_idx, len(col) + 2)
            
        # Add autofilter to all columns
        worksheet.autofilter(0, 0, df.shape[0], df.shape[1] - 1)

    return filename

def format_to_hyperlinks(paths):
    """
    Recibe una lista de rutas (strings) y devuelve una nueva lista
    con la fórmula Excel HYPERLINK de dos parámetros:
      =HYPERLINK("ruta", "nombre_de_archivo")
    
    El "nombre_de_archivo" se extrae de la ruta sin extensión.
    
    Ejemplo:
        >>> paths = ["C:/img/a.png", "C:/docs/report.pdf"]
        >>> format_to_hyperlinks(paths)
        [
          '=HYPERLINK("C:/img/a.png", "a")',
          '=HYPERLINK("C:/docs/report.pdf", "report")'
        ]
    """
    links = []
    for ruta in paths:
        # basename extrae "a.png", splitext lo separa en ("a", ".png")
        nombre_archivo = os.path.splitext(os.path.basename(ruta))[0]
        # construye la fórmula con ruta y nombre de archivo
        links.append(f'=HYPERLINK("{ruta}", "{nombre_archivo}")')
    return links

def normalize_dict_lengths(informacion):
    """
    Toma un dict cuyos valores son listas de longitudes posiblemente diferentes,
    y devuelve un nuevo dict donde todas las listas se han rellenado al tamaño
    de la más larga con el string " " (espacio).

    Ejemplo:
        informacion = {
            "Columna 1": ["A", "B", "C", "D"],
            "Columna 2": ["X", "Y"],
            "Columna 3": ["Z", "W", "V"]
        }
        => normalize_dict_lengths(informacion) será:
        {
            "Columna 1": ["A", "B", "C", "D"],
            "Columna 2": ["X", "Y", " ", " "],
            "Columna 3": ["Z", "W", "V", " "]
        }
    """
    # Determina la longitud máxima entre todas las listas
    max_len = max(len(lst) for lst in informacion.values())

    # Construye un nuevo diccionario con listas rellenadas
    padded = {}
    for key, lst in informacion.items():
        # calcula cuántos espacios hacer
        padding = [" "] * (max_len - len(lst))
        padded[key] = lst + padding

    return padded

def get_file_count(folder_path: str) -> int:
    """
    Returns the total number of files in the given folder, including all subdirectories.
    
    Args:
        folder_path: Path to the folder you want to count files in.
        
    Returns:
        The number of files found.
    """
    return sum(len(files) for _, _, files in os.walk(folder_path))