import time
import json
import shutil
from pathlib import Path
from hivemind import HiveMind
from typing import List, Dict
import cv2
import numpy as np
from logconfig import logger


def run_bio_inspired_frame_selection(input_dir: str, output_dir: str,
                                     n_scouts: int = 20, n_workers: int = 40) -> Dict:
    """
    Input: input_dir (str), output_dir (str), n_scouts (int), n_workers (int)
    Context: Main function to execute bio-inspired frame selection.
    Output: Dictionary with selection summary
    """

    start_time = time.time()

    # Cargar frames desde input_dir
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = list(Path(input_dir).glob('*.*'))
    image_files = [p for p in all_files if p.suffix.lower() in valid_exts]
    image_files.sort(key=lambda x: x.name)
    logger.info(f"Cargando {len(image_files)} imágenes desde '{input_dir}'")

    frames: List[np.ndarray] = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"No se pudo leer la imagen: {img_path}")
            continue
        frames.append(img)
    if not frames:
        logger.error("No se cargó ninguna imagen válida. Terminando ejecución.")
        return {}

    # Ejecutar enjambre bio-inspirado
    hive = HiveMind(n_scouts=n_scouts, n_workers=n_workers)
    selected_indices = hive.foraging_cycle(frames)
    logger.info(f"Índices seleccionados: {selected_indices}")

    # Preparar carpeta de salida y subcarpetas
    out_base = Path(output_dir)
    images_dir = out_base / 'images'
    data_dir = out_base / 'data'
    out_base.mkdir(parents=True, exist_ok=True)

    # Limpiar y recrear subcarpetas
    for d in (images_dir, data_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Limpia carpetas '{images_dir}' y '{data_dir}'")

    # Copiar las imágenes seleccionadas a output/images
    selected_names: List[str] = []
    for idx in selected_indices:
        try:
            src = image_files[idx]
            dst = images_dir / src.name
            shutil.copy(src, dst)
            selected_names.append(src.name)
        except IndexError:
            logger.warning(f"Índice fuera de rango al copiar: {idx}")

    # Guardar resumen en JSON dentro de output/data
    elapsed = time.time() - start_time
    summary = {
        'selected_indices': selected_indices,
        'selected_frames': selected_names,
        'n_scouts': n_scouts,
        'n_workers': n_workers,
        'execution_time_s': elapsed
    }
    summary_path = data_dir / 'selection_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Resumen de selección guardado en '{summary_path}'")

    return summary

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    # Importa la función principal (asume que está en el mismo archivo o en un módulo llamado `bio_frame_selector`)
    # from bio_frame_selector import run_bio_inspired_frame_selection

    parser = argparse.ArgumentParser(
        description="Ejecuta la selección bio-inspirada de frames para SFM"
    )
    parser.add_argument(
        "un_numero",
        type=str,
        help="Identificador de la carpeta base (p.ej. '001', '002', etc.)"
    )
    parser.add_argument(
        "--n_scouts",
        type=int,
        default=20,
        help="Número de abejas exploradoras (por defecto: 20)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=40,
        help="Número de abejas trabajadoras (por defecto: 40)"
    )
    args = parser.parse_args()

    # Construye rutas basadas en la variable UN_NUMERO
    base = args.un_numero
    input_dir = Path(f"{base}/images/data")
    output_dir = Path(f"{base}/output")

    # Ejecuta la selección de frames
    summary = run_bio_inspired_frame_selection(
        str(input_dir),
        str(output_dir),
        n_scouts=args.n_scouts,
        n_workers=args.n_workers
    )

    # Muestra el resumen en pantalla
    print(json.dumps(summary, indent=2, ensure_ascii=False))