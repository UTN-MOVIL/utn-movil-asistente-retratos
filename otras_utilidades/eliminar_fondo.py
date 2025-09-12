import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # …/funcionalidades_validador_retratos
sys.path.insert(0, str(ROOT))
from modulos.eliminacion_de_fondo import FUN_OBTENER_IMAGEN_SIN_FONDO

# Extensiones de imagen aceptadas
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def procesar_directorio(dir_origen: Path, dir_destino: Path, recursivo: bool = False) -> None:
    """
    Aplica FUN_OBTENER_IMAGEN_SIN_FONDO a todas las imágenes en dir_origen
    y guarda PNGs con fondo eliminado en dir_destino.
    """
    if not dir_origen.exists():
        print(f"❌ No existe el directorio de origen: {dir_origen}")
        return

    dir_destino.mkdir(parents=True, exist_ok=True)

    # Recolectar archivos
    iter_paths = dir_origen.rglob("*") if recursivo else dir_origen.glob("*")
    archivos = [p for p in iter_paths if p.is_file() and p.suffix.lower() in EXTS]
    if not archivos:
        print(f"ℹ️ No se encontraron imágenes en: {dir_origen}")
        return

    total = len(archivos)
    ok = 0

    print(f"🔎 Procesando {total} archivo(s) desde: {dir_origen}")
    for idx, ruta in enumerate(sorted(archivos), start=1):
        # Leer imagen
        imagen_entrada = cv2.imread(str(ruta))
        if imagen_entrada is None:
            print(f"[{idx}/{total}] ⚠️ No se pudo leer: {ruta.name}")
            continue

        # Procesar
        try:
            imagen_sin_fondo = FUN_OBTENER_IMAGEN_SIN_FONDO(imagen_entrada)
        except Exception as e:
            print(f"[{idx}/{total}] ❌ Error procesando {ruta.name}: {e}")
            continue

        if imagen_sin_fondo is None:
            print(f"[{idx}/{total}] ⚠️ La función devolvió None para: {ruta.name}")
            continue

        # Construir nombre de salida (PNG)
        salida = dir_destino / f"{ruta.stem}.png"

        # Evitar colisiones de nombre
        contador = 1
        while salida.exists():
            salida = dir_destino / f"{ruta.stem}_sin_fondo_{contador}.png"
            contador += 1

        # Guardar
        if cv2.imwrite(str(salida), imagen_sin_fondo):
            ok += 1
            print(f"[{idx}/{total}] ✅ {ruta.name} -> {salida.name}")
        else:
            print(f"[{idx}/{total}] ❌ No se pudo guardar: {salida}")

    print(f"🏁 Listo. Guardadas {ok}/{total} imágenes en: {dir_destino}")

if __name__ == "__main__":
    # ⇩⇩⇩ EDITA ESTAS RUTAS ⇩⇩⇩
    DIR_ORIGEN = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\FRAMES_GIROS\RECORTADOS")
    DIR_DESTINO = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\FRAMES_GIROS\SIN_FONDO")
    # ⇧⇧⇧ EDITA ESTAS RUTAS ⇧⇧⇧

    # Cambia recursivo=True si quieres incluir subcarpetas
    procesar_directorio(DIR_ORIGEN, DIR_DESTINO, recursivo=False)
