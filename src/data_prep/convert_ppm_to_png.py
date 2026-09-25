"""Utilitário para converter imagens PPM do dataset GTSRB para formato PNG.
Isso remove o gargalo de I/O causado pela leitura lenta de PPM através do Pillow,
permitindo a decodificação nativa e paralelizada no tf.data usando C++.
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

try:
    from PIL import Image
except ImportError:
    Image = None

def convert_single_ppm(ppm_path: Path, delete_original: bool = False) -> bool:
    png_path = ppm_path.with_suffix('.png')
    try:
        with Image.open(ppm_path) as img:
            img.save(png_path, "PNG")
        if delete_original:
            ppm_path.unlink()
        return True
    except Exception as e:
        print(f"Erro ao converter {ppm_path}: {e}")
        return False

def main():
    if Image is None:
        print("Erro: Pillow (PIL) não está instalado. Instale com `pip install Pillow`.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Converte imagens PPM do GTSRB para PNG.")
    parser.add_argument("source_dir", type=Path, help="Diretório contendo os arquivos PPM (ex: datasets/gtsrb).")
    parser.add_argument("--delete", action="store_true", help="Remove os arquivos PPM originais após conversão.")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Número de processos paralelos.")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Erro: O diretório '{args.source_dir}' não existe.", file=sys.stderr)
        return 1

    ppm_files = list(args.source_dir.rglob("*.ppm"))
    if not ppm_files:
        print(f"Nenhum arquivo .ppm encontrado em '{args.source_dir}'.", file=sys.stderr)
        return 0

    print(f"Iniciando conversão de {len(ppm_files)} arquivos .ppm usando {args.workers} workers...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(convert_single_ppm, ppm_files, [args.delete]*len(ppm_files))
        for res in results:
            if res:
                success_count += 1
                
    print(f"Conversão concluída: {success_count}/{len(ppm_files)} imagens convertidas com sucesso.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
