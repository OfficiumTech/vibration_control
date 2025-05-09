import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

def modelo_ppv(a, b, c, Q, D):
    """
    Modelo da equação de atenuação: v = a * (Q^b) / (D^c)
    """
    return a * (Q ** b) / (D ** c)

def calcular_regressao(ui):
    """
    Calcula a regressão não-linear para a equação de atenuação:
    v = a * (Q^b) / (D^c), usando apenas NumPy (sem Scipy).
    Realiza uma busca iterativa por mínimos quadrados.
    """
    try:
        # Verifica se há resultados disponíveis na aba PROCESS
        resultados = ui.resultados_processados
        if not resultados:
            raise Exception("Nenhum resultado encontrado. Execute o processamento primeiro.")

        # Extrai os dados necessários
        distancias = np.array([r["Distância (m)"] for r in resultados], dtype=float)
        cargas = np.array([r["Carga (KG)"] for r in resultados], dtype=float)
        ppvs = np.array([r["PPV/PVS"] for r in resultados], dtype=float)

        # Aplica máscara para garantir que os dados são positivos e válidos
        mascara = (
            (cargas > 0) & np.isfinite(cargas) &
            (distancias > 0) & np.isfinite(distancias) &
            (ppvs > 0) & np.isfinite(ppvs)
        )

        # Aplica a máscara
        cargas = cargas[mascara]
        distancias = distancias[mascara]
        ppvs = ppvs[mascara]

        if len(cargas) < 3:
            raise Exception("Não há dados válidos suficientes para calcular a equação.")

        # Busca global bruta (grid search)
        melhor_erro = float("inf")
        melhor_a, melhor_b, melhor_c = 1.0, 1.0, 1.0

        # Intervalo de busca inicial para b e c (expoentes)
        b_vals = np.linspace(0.1, 2.5, 30)
        c_vals = np.linspace(0.1, 2.5, 30)

        for b in b_vals:
            for c in c_vals:
                # Estima o melhor 'a' para cada par (b, c) via mínimos quadrados
                X = (cargas ** b) / (distancias ** c)
                a = np.sum(ppvs * X) / np.sum(X ** 2)

                # Calcula PPV estimado e erro médio quadrático
                ppv_estimado = modelo_ppv(a, b, c, cargas, distancias)
                erro = np.mean((ppvs - ppv_estimado) ** 2)

                if erro < melhor_erro:
                    melhor_erro = erro
                    melhor_a, melhor_b, melhor_c = a, b, c

        # Refinamento local em torno dos melhores valores encontrados
        passos_finos = 10
        delta_b = 0.1
        delta_c = 0.1

        b_range = np.linspace(melhor_b - delta_b, melhor_b + delta_b, passos_finos)
        c_range = np.linspace(melhor_c - delta_c, melhor_c + delta_c, passos_finos)

        for b in b_range:
            for c in c_range:
                X = (cargas ** b) / (distancias ** c)
                a = np.sum(ppvs * X) / np.sum(X ** 2)
                ppv_estimado = modelo_ppv(a, b, c, cargas, distancias)
                erro = np.mean((ppvs - ppv_estimado) ** 2)

                if erro < melhor_erro:
                    melhor_erro = erro
                    melhor_a, melhor_b, melhor_c = a, b, c

        # Coeficientes finais
        a, b, c = melhor_a, melhor_b, melhor_c
        ppv_estimado = modelo_ppv(a, b, c, cargas, distancias)

        # Cálculo do R²
        ss_res = np.sum((ppvs - ppv_estimado) ** 2)
        ss_tot = np.sum((ppvs - np.mean(ppvs)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Exibe os resultados na interface
        texto = (
            "Equação de Atenuação:\n"
            f"v = {a:.4f} * (Q^{b:.4f}) / (D^{c:.4f})\n\n"
            f"Coeficiente de determinação (R²): {r2:.4f}\n"
        )
        ui.textBrowser_atenuacao.setText(texto)

        # Armazena os parâmetros para exportações futuras
        ui.parametros_regressao = a, b, c

        # Gera gráfico comparativo
        gerar_grafico_ajuste(distancias, cargas, ppvs, a, b, c)

    except Exception as e:
        QMessageBox.critical(None, "Erro na regressão", str(e))


def gerar_grafico_ajuste(distancias, cargas, ppvs, a, b, c):
    """
    Gera um gráfico comparando os valores observados de PPV com os estimados.
    """
    try:
        ppv_estimado = modelo_ppv(a, b, c, cargas, distancias)

        plt.figure(figsize=(4, 3))
        plt.scatter(ppvs, ppv_estimado, color='blue', label="Estimado vs Observado")
        plt.plot([min(ppvs), max(ppvs)], [min(ppvs), max(ppvs)], 'r--', label="Ideal (y=x)")
        plt.xlabel("PPV Observado (mm/s)")
        plt.ylabel("PPV Estimado (mm/s)")
        plt.title("Comparação PPV Observado x Estimado")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        QMessageBox.critical(None, "Erro no gráfico", str(e))
