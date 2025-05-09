from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

def modelo_atenuacao(xdata, a, b, c):
    Q, D = xdata
    #return a * (Q ** b) / (D ** c)
    return a * (Q ** b) * (D ** c)

def calcular_regressao(ui):
    try:
        try:
            resultados = ui.resultados_processados
            distancias = np.array([r["Distância (m)"] for r in resultados], dtype=float)
            cargas = np.array([r["Carga (KG)"] for r in resultados], dtype=float)
            ppvs = np.array([r["PPV/PVS"] for r in resultados], dtype=float)
        except:
            QMessageBox.information(None, "Regressão - Blaster Vibration Control", "Execute o processamento primeiro!")

        # Filtrar valores válidos
        mascara = (
            (cargas > 0) & np.isfinite(cargas) &
            (distancias > 0) & np.isfinite(distancias) &
            (ppvs > 0) & np.isfinite(ppvs)
        )

        cargas = cargas[mascara]
        distancias = distancias[mascara]
        ppvs = ppvs[mascara]

        if len(cargas) < 3:
            raise Exception("Não há dados válidos suficientes para calcular a equação.")

        # Regressão não linear
        xdata = (cargas, distancias)
        ydata = ppvs

        p0 = [1.0, 1.0, 1.0]  # chute inicial
        #p0 = [0.1, 0.1, 0.1]
        #popt, _ = curve_fit(modelo_atenuacao, xdata, ydata, p0=p0, maxfev=10000)
        # Ajuste com controle de tolerância
        popt, pcov = curve_fit(modelo_atenuacao, xdata=(cargas, distancias), ydata=ppvs, p0=p0,
            method='lm',  # Trust Region Reflective: method='trf' / Levenberg-Marquardt: method='lm'
            ftol=1e-10,    # Tolerância para a função objetivo (erro)
            xtol=1e-10,    # Tolerância para os parâmetros
            gtol=1e-10,    # Tolerância para o gradiente
            maxfev=10000   # Número máximo de avaliações da função
        )
        
        a, b, c = popt
        perr = np.sqrt(np.diag(pcov)) # Erros padrão 
        #a_err, b_err, c_err = perr

        # Calcular R²
        ppv_estimado = modelo_atenuacao(xdata, a, b, c)
        ss_res = np.sum((ydata - ppv_estimado) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Exibir resultados
        texto = (
            "Equação de Atenuação:\n"
            f"v = {a:.4f} * (Q^{b:.4f}) * (D^{c:.4f})\n\n"
            f"Coeficiente de determinação (R²): {r2:.4f}\n"
            f"a = {a:.4f} ± {perr[0]:.4f}\n"
            f"b = {b:.4f} ± {perr[1]:.4f}\n"
            f"c = {c:.4f} ± {perr[2]:.4f}"
        )
        ui.textBrowser_atenuacao.setText(texto)
        ui.parametros_regressao = a, b, c

        #gerar_grafico_ajuste(distancias, cargas, ppvs, a, b, c)
        ui._dados_regressao = (distancias, cargas, ppvs, a, b, c)

    except Exception as e:
        QMessageBox.critical(None, "Erro na regressão", str(e))


def gerar_grafico_ajuste_ui(ui):
        try:
            dados = ui._dados_regressao
            gerar_grafico_ajuste(*dados)
        except Exception as e:
            QMessageBox.critical(None, "Erro ao gerar gráfico", "Execute a regressão primeiro.")


def gerar_grafico_ajuste(distancias, cargas, ppvs, a, b, c):
    try:
        v_estimado = a * (cargas ** b) * (distancias ** c)

        plt.figure(figsize=(4, 3))
        plt.scatter(ppvs, v_estimado, color='blue', label="Estimado vs Observado")
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
