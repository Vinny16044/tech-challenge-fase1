"""
Tech Challenge Fase 1 - CLI Principal
Responsável: Vinicius (P.O. e Desenvolvedor)

Ponto de entrada para execução do projeto via linha de comando.
Uso: python main.py [--dataset seer|wisconsin|all] [--no-figures] [--no-models]
"""

import argparse
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from pipeline import run_seer_pipeline, run_wisconsin_pipeline, run_full_pipeline
from config import ensure_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Tech Challenge Fase 1 - Sistema de Suporte ao Diagnóstico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                        # Executa pipeline completo (ambos datasets)
  python main.py --dataset seer         # Apenas dataset SEER Clínico
  python main.py --dataset wisconsin    # Apenas dataset Wisconsin Tumoral
  python main.py --no-figures           # Sem gerar gráficos
  python main.py --no-models            # Sem salvar modelos em disco
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        choices=['seer', 'wisconsin', 'all'],
        default='all',
        help='Dataset a processar (default: all)'
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Não gerar gráficos/figuras'
    )
    parser.add_argument(
        '--no-models',
        action='store_true',
        help='Não salvar modelos treinados em disco'
    )

    args = parser.parse_args()

    ensure_dirs()

    save_fig = not args.no_figures
    save_mod = not args.no_models

    if args.dataset == 'seer':
        run_seer_pipeline(save_figures=save_fig, save_models=save_mod)
    elif args.dataset == 'wisconsin':
        run_wisconsin_pipeline(save_figures=save_fig, save_models=save_mod)
    else:
        run_full_pipeline(save_figures=save_fig, save_models=save_mod)


if __name__ == '__main__':
    main()
