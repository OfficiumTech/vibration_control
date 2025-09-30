from PyQt5.QtWidgets import QMessageBox
from scipy.optimize import curve_fit
import numpy as np
import traceback
import matplotlib.pyplot as plt


def modelo_atenuacao(xdata, a, b, c):
    Q, D = xdata
    return a * (Q ** b) * (D ** c)

def log(mensagem):
    QMessageBox.information(None, "Regressão - Blaster Vibration Control", mensagem)

def calcular_regressao(ui):
    try:
        try:
            resultados = ui.resultados_processados
            distancias = np.array([r["Distância (m)"] for r in resultados], dtype=float)
            cargas = np.array([r["Carga (KG)"] for r in resultados], dtype=float)
            ppvs = np.array([r["PPV/PVS"] for r in resultados], dtype=float)
        except:
            log("Execute o processamento primeiro!")
            return

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
            log("Não há dados válidos suficientes para calcular a equação.")
            return

        if ui.checkBox_levenberg.isChecked():
            # ---------- MÉTODO LEVENBERG-MARQUARDT ----------
            xdata = (cargas, distancias)
            ydata = ppvs
            try:
                ai = float(ui.lineEdit_a0.text())
                bi = float(ui.lineEdit_b0.text())
                ci = float(ui.lineEdit_c0.text())
                er = float(ui.lineEdit_erro.text())
                p0 = [ai, bi, ci]
            except Exception as e:
                p0 = [0.1, 0.1, 0.1]
                er = 1e-10
                log(f"Erro nos campos a0, b0, c0 ou erro, usando valores padrão.\nERRO:\n{e}")
            
            popt, pcov = curve_fit(modelo_atenuacao, xdata=(cargas, distancias), ydata=ppvs, p0=p0,
                method='lm', # Trust Region Reflective: method='trf' / Levenberg-Marquardt: method='lm'
                ftol=er,    # Tolerância para a função objetivo (erro)
                xtol=er,    # Tolerância para os parâmetros
                gtol=er,    # Tolerância para o gradiente
                maxfev=10000    # Número máximo de avaliações da função
            )

            a, b, c = popt
            perr = np.sqrt(np.diag(pcov))
            # Calcular R²
            ppv_estimado = modelo_atenuacao(xdata, a, b, c)

        else:
            # ------------------- Linear (log-log) com numpy -------------------
            log_q = np.log10(cargas)
            log_d = np.log10(distancias)
            log_ppv = np.log10(ppvs)

            X = np.column_stack([np.ones(len(log_q)), log_q, log_d])  # [1, log(Q), log(D)]
            coef, _, _, _ = np.linalg.lstsq(X, log_ppv, rcond=None)

            log_a, b, c = coef
            a = 10 ** log_a
            ppv_estimado = a * (cargas ** b) * (distancias ** c)
            perr = [0, 0, 0]  # Erros não disponíveis nessa abordagem

        # Calcular R²
        ss_res = np.sum((ppvs - ppv_estimado) ** 2)
        ss_tot = np.sum((ppvs - np.mean(ppvs)) ** 2)
        r2 = 1 - ss_res / ss_tot


        # Exibir resultados--------------------------------
        texto = (
            "Equação de Atenuação:\n"
            f"v = {a:.4f} * (Q^{b:.4f}) * (D^{c:.4f})\n\n"
            f"Coeficiente de determinação (R²): {r2:.4f}\n"
        )

        # Adicionar cada linha apenas se o erro for um número finito e diferente de zero
        def linha_param(nome, valor, erro):
            if erro is not None and np.isfinite(erro) and erro != 0:
                return f"{nome} = {valor:.4f} ± {erro:.4f}\n"
            else:
                return f"{nome} = {valor:.4f}\n"

        texto += linha_param("a", a, perr[0])
        texto += linha_param("b", b, perr[1])
        texto += linha_param("c", c, perr[2])
        # Fim exibir resul-------------------------------------

        ui.textBrowser_atenuacao.setText(texto)
        ui.parametros_regressao = a, b, c
        ui._dados_regressao = (distancias, cargas, ppvs, a, b, c)

    except Exception as e:
        tb = traceback.format_exc()
        log(f"Erro na regressão. ERRO:\n{e}\nTraceback:\n{tb}")



def gerar_grafico_ajuste_ui(ui):
        try:
            dados = ui._dados_regressao
            gerar_grafico_ajuste(*dados)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Erro ao gerar gráfico, Execute a regressão primeiro.\n {e}\n {tb}")
            


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