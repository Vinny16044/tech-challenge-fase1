"""
Tech Challenge - CLI Principal
Responsável: Vinicius (P.O. e Desenvolvedor)

Ponto de entrada para execução do projeto via linha de comando.
Uso: python main.py [--dataset seer|wisconsin|all] [--no-figures] [--no-models]
                    [--explain] [--llm-model MODELO]
"""

import argparse
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from pipeline import run_seer_pipeline, run_wisconsin_pipeline
from config import ensure_dirs


def _explicar_resultado(result: dict, llm_model: str):
    """
    Usa a camada de LLM (Vinicius) para gerar uma explicação em linguagem
    natural da predição do melhor modelo sobre uma amostra de teste.
    """
    from llm.interpreter import LLMInterpreter, build_patient_case_from_pipeline

    best_name = result['best_model'][0]
    interpreter = LLMInterpreter(model=llm_model)
    modo = "Gemini" if interpreter.is_llm_available() else "offline (sem GEMINI_API_KEY)"

    print("\n" + "=" * 70)
    print(f"INTERPRETAÇÃO VIA LLM — {result['dataset']}  [modo: {modo}]")
    print("=" * 70)

    caso = build_patient_case_from_pipeline(result, best_name, sample_index=0)
    print(f"\n>>> Explicação da predição (melhor modelo: {best_name})\n")
    print(interpreter.explain_prediction(caso))


def main():
    parser = argparse.ArgumentParser(
        description='Tech Challenge - Sistema de Suporte ao Diagnóstico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                        # Pipeline completo (ambos datasets)
  python main.py --dataset seer         # Apenas dataset SEER Clínico
  python main.py --dataset wisconsin    # Apenas dataset Wisconsin Tumoral
  python main.py --no-figures           # Sem gerar gráficos
  python main.py --no-models            # Sem salvar modelos em disco
  python main.py --dataset wisconsin --explain   # + explicação via LLM (Gemini)

Para a explicação via LLM real, defina a chave gratuita do Gemini:
  Linux/Mac:  export GEMINI_API_KEY="sua-chave"
  Windows:    set GEMINI_API_KEY=sua-chave
  Sem a chave, a explicação é gerada em modo offline determinístico.
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        choices=['seer', 'wisconsin', 'all'],
        default='all',
        help='Dataset a processar (default: all)'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Não gerar gráficos/figuras'
    )
    parser.add_argument(
        '--no-models', action='store_true',
        help='Não salvar modelos treinados em disco'
    )
    parser.add_argument(
        '--explain', action='store_true',
        help='Gerar explicação em linguagem natural (LLM) ao final do pipeline'
    )
    parser.add_argument(
        '--llm-model', default='gemini-flash-latest',
        help='Modelo da LLM Gemini (default: gemini-flash-latest)'
    )

    args = parser.parse_args()

    ensure_dirs()
    save_fig = not args.no_figures
    save_mod = not args.no_models

    resultados = []
    if args.dataset in ('seer', 'all'):
        resultados.append(run_seer_pipeline(save_figures=save_fig, save_models=save_mod))
    if args.dataset in ('wisconsin', 'all'):
        resultados.append(run_wisconsin_pipeline(save_figures=save_fig, save_models=save_mod))

    if args.explain:
        for result in resultados:
            try:
                _explicar_resultado(result, args.llm_model)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[Aviso] Não foi possível gerar a explicação via LLM: {exc}")


if __name__ == '__main__':
    main()
