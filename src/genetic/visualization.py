"""
Visualizações dos experimentos da Fase 2.
Responsável: Paola
"""

import os
import matplotlib.pyplot as plt


def plot_baseline_vs_optimized(
    df_comparison,
    metric: str = "recall",
    output_dir: str = "reports/figures/fase2"
):
    os.makedirs(output_dir, exist_ok=True)

    baseline_col = f"baseline_{metric}"
    optimized_col = f"optimized_{metric}"

    if baseline_col not in df_comparison.columns:
        raise ValueError(f"Coluna não encontrada: {baseline_col}")

    if optimized_col not in df_comparison.columns:
        raise ValueError(f"Coluna não encontrada: {optimized_col}")

    df_plot = df_comparison.copy()

    # Evita repetir o mesmo modelo várias vezes quando houver mais de um experimento
    df_plot = df_plot.sort_values(optimized_col, ascending=False)
    df_plot = df_plot.drop_duplicates(subset=["model"], keep="first")

    labels = df_plot["model"].tolist()
    baseline_values = df_plot[baseline_col].tolist()
    optimized_values = df_plot[optimized_col].tolist()

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - width / 2 for i in x], baseline_values, width, label="Original")
    ax.bar([i + width / 2 for i in x], optimized_values, width, label="Otimizado")

    ax.set_title(f"Comparação Original vs. Otimizado - {metric.upper()}")
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"baseline_vs_optimized_{metric}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo em: {output_path}")

    return output_path