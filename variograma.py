#Versão3============================================================================================
# -*- coding: utf-8 -*-
# variograma.py

#=========MODIFICADO COM AUTO FIT========================================================================================
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit


class VariogramaWidget(QtWidgets.QDialog):
    parametros_confirmados = QtCore.pyqtSignal(float, float, float, str, float, float)  # inclui scale

    def ajustar_power(self):
        """Ajusta scale e exponent ao variograma empírico quando modelo = power."""
        def power_model(h, scale, exponent):
            return self.nugget_input.value() + scale * (h ** exponent)
        try:
            # pega o valor do range definido na interface
            max_range = self.range_input.value()
            
            # filtra os bins que estão dentro do range
            mask = self.bin_centers <= max_range
            h_filtrado = self.bin_centers[mask]
            semi_filtrado = self.semi_media[mask]
            
            popt, _ = curve_fit(power_model,
                                h_filtrado, #self.bin_centers,
                                semi_filtrado, #self.semi_media,
                                p0=[1.0, 1.5],  # chute inicial scale, exponent
                                bounds=([0.0, 0.1], [np.inf, 3.0]))
            scale, exponent = popt
            self.exponent_input.setValue(exponent)  # atualiza GUI
            self.scale = scale
            self.plotar_variograma()  # plota já ajustado
        except Exception as e:
            print(f"⚠️ Falha no ajuste power: {e}")
            self.scale = 1.0

    def __init__(self, coords, valores):
        super().__init__()
        self.coords = coords
        self.valores = valores
        self.scale = 1.0  # Valor inicial - usado apenas para power

        self.modelos = ["spherical", "exponential", "gaussian", "linear", "power"]

        self.setWindowTitle("Ajuste do Semivariograma")
        self.resize(600, 550)

        self.init_ui()
        self.calcular_variograma_empirico()
        self.definir_parametros_iniciais()
        self.plotar_variograma()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        form = QtWidgets.QFormLayout()
        self.nugget_input = QtWidgets.QDoubleSpinBox()
        self.sill_input = QtWidgets.QDoubleSpinBox()
        self.range_input = QtWidgets.QDoubleSpinBox()
        self.modelo_input = QtWidgets.QComboBox()
        self.modelo_input.addItems(self.modelos)
        self.exponent_input = QtWidgets.QDoubleSpinBox()

        for w in [self.nugget_input, self.sill_input, self.range_input, self.exponent_input]:
            w.setRange(0.0, 1e6)
            w.setDecimals(4)
            w.setSingleStep(0.1)

        self.exponent_input.setValue(1.95)
        self.exponent_input.setToolTip("Expoente usado apenas no modelo 'power'")
        self.exponent_input.setVisible(False)

        form.addRow("Nugget:", self.nugget_input)
        form.addRow("Sill:", self.sill_input)
        form.addRow("Range:", self.range_input)
        form.addRow("Modelo:", self.modelo_input)
        form.addRow("Expoente:", self.exponent_input)
        layout.addLayout(form)

        self.modelo_input.currentTextChanged.connect(self.alternar_modelo)

        botoes = QtWidgets.QHBoxLayout()
        self.botao_atualizar = QtWidgets.QPushButton("Atualizar Gráfico")
        self.botao_auto_fit = QtWidgets.QPushButton("Auto Fit")
        self.botao_ok = QtWidgets.QPushButton("OK")
        botoes.addWidget(self.botao_atualizar)
        botoes.addWidget(self.botao_auto_fit)
        botoes.addWidget(self.botao_ok)
        layout.addLayout(botoes)

        self.botao_atualizar.clicked.connect(self.plotar_variograma)
        self.botao_auto_fit.clicked.connect(self.ajustar_power)
        self.botao_ok.clicked.connect(self.confirmar_parametros)

        self.botao_auto_fit.setVisible(False)  # aparece só para power

    def alternar_modelo(self, texto):
        power = texto == "power"
        self.exponent_input.setVisible(power)
        self.botao_auto_fit.setVisible(power)
        self.sill_input.setEnabled(not power)  # sill não editável para power
        self.sill_input.setToolTip("Sill não usado no modelo power")

    def calcular_variograma_empirico(self):
        dist = pdist(self.coords)
        diff = pdist(self.valores[:, None], metric='euclidean')
        semi = 0.5 * diff**2
        
        x = np.array([p[0] for p in self.coords]) #Extrai coord x
        dx = np.min(np.diff(np.unique(x))) #Calcular a menor distancia entre elas
        
        maxDist = np.max(dist)
        malha = dx #20 #tamanho da malha de pontos simulados 20x20m
        fator_ajuste = 15 #Valor por testes
        nbins = round((maxDist/(125*malha))*fator_ajuste) #heurística para definir a quantidade de bins
        if nbins < 8:
            nbins = 8
        if nbins > 30:
            nbins = 30
        
        bins = np.linspace(0, maxDist, nbins+1 )
        #bins = np.linspace(0, np.max(dist), 15)
        indices = np.digitize(dist, bins)

        self.bin_centers = []
        self.semi_media = []
        for i in range(1, len(bins)):
            mask = indices == i
            if np.any(mask):
                self.bin_centers.append(np.mean(dist[mask]))
                self.semi_media.append(np.mean(semi[mask]))

        self.bin_centers = np.array(self.bin_centers)
        self.semi_media = np.array(self.semi_media)

    def definir_parametros_iniciais(self):
        nugget = min(self.semi_media)
        sill = max(self.semi_media)
        range_ = max(self.bin_centers) * 0.8

        self.nugget_input.setValue(nugget)
        self.sill_input.setValue(sill)
        self.range_input.setValue(range_)

    def modelo_variograma(self, h, nugget, sill, range_, modelo, exponent=1.95):
        h = np.array(h)
        if modelo == "spherical":
            return np.where(h <= range_,
                            nugget + (sill - nugget)*(1.5*(h/range_) - 0.5*(h/range_)**3),
                            sill)
        elif modelo == "exponential":
            return nugget + (sill - nugget)*(1 - np.exp(-h / (range_ / 3)))
        elif modelo == "gaussian":
            return nugget + (sill - nugget)*(1 - np.exp(-(h**2) / ((range_/3)**2)))
        elif modelo == "linear":
            return np.minimum(nugget + (sill - nugget)*(h / range_), sill)
        elif modelo == "power":
            return nugget + self.scale * (h ** exponent)
        else:
            return np.full_like(h, nugget)

    def plotar_variograma(self):
        nugget = self.nugget_input.value()
        sill = self.sill_input.value()
        range_ = self.range_input.value()
        modelo = self.modelo_input.currentText()
        exponent = self.exponent_input.value()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(self.bin_centers, self.semi_media, label="Empírico", color="blue")

        h_vals = np.linspace(0, max(self.bin_centers)*1.1, 100)
        gamma_vals = self.modelo_variograma(h_vals, nugget, sill, range_, modelo, exponent)

        ax.plot(h_vals, gamma_vals, label=f"Modelo: {modelo}", color="red")
        ax.set_xlabel("Distância (h)")
        ax.set_ylabel("Semivariância γ(h)")
        ax.legend()
        self.canvas.draw()

    def confirmar_parametros(self):
        nugget = self.nugget_input.value()
        sill = self.sill_input.value()
        range_ = self.range_input.value()
        modelo = self.modelo_input.currentText()
        exponent = self.exponent_input.value()

        # scale só faz sentido para power
        scale = self.scale if modelo == "power" else 1.0

        self.parametros_confirmados.emit(nugget, sill, range_, modelo, exponent, scale)
        self.accept()


def ajustar_variograma(coords, valores):
    widget = VariogramaWidget(coords, valores)
    resultado = {}

    def salvar_parametros(n, s, r, m, e, sc):
        resultado["nugget"] = n
        resultado["sill"] = s
        resultado["range"] = r
        resultado["modelo"] = m
        resultado["exponent"] = e
        resultado["scale"] = sc

    widget.parametros_confirmados.connect(salvar_parametros)

    if widget.exec_() == QtWidgets.QDialog.Accepted:
        return (resultado.get("nugget"),
                resultado.get("sill"),
                resultado.get("range"),
                resultado.get("modelo"),
                resultado.get("exponent"),
                resultado.get("scale"))
    else:
        return None, None, None, None, None, None




'''#=========ANTIGO========================================================================================
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial.distance import pdist


class VariogramaWidget(QtWidgets.QDialog):
    parametros_confirmados = QtCore.pyqtSignal(float, float, float, str, float)  # + exponent

    def __init__(self, coords, valores):
        super().__init__()
        self.coords = coords
        self.valores = valores

        self.modelos = ["spherical", "exponential", "gaussian", "linear", "power"]

        self.setWindowTitle("Ajuste do Semivariograma")
        self.resize(600, 550)

        self.init_ui()
        self.calcular_variograma_empirico()
        self.definir_parametros_iniciais()
        self.plotar_variograma()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        form = QtWidgets.QFormLayout()
        self.nugget_input = QtWidgets.QDoubleSpinBox()
        self.sill_input = QtWidgets.QDoubleSpinBox()
        self.range_input = QtWidgets.QDoubleSpinBox()
        self.modelo_input = QtWidgets.QComboBox()
        self.modelo_input.addItems(self.modelos)
        self.exponent_input = QtWidgets.QDoubleSpinBox()

        for w in [self.nugget_input, self.sill_input, self.range_input, self.exponent_input]:
            w.setRange(0.0, 1e6)
            w.setDecimals(4)
            w.setSingleStep(0.1)

        self.exponent_input.setValue(1.95)
        self.exponent_input.setToolTip("Expoente usado apenas no modelo 'power'")
        self.exponent_input.setVisible(False)

        form.addRow("Nugget:", self.nugget_input)
        form.addRow("Sill:", self.sill_input)
        form.addRow("Range:", self.range_input)
        form.addRow("Modelo:", self.modelo_input)
        form.addRow("Expoente:", self.exponent_input)
        layout.addLayout(form)

        self.modelo_input.currentTextChanged.connect(self.alternar_expoente)

        botoes = QtWidgets.QHBoxLayout()
        self.botao_atualizar = QtWidgets.QPushButton("Atualizar Gráfico")
        self.botao_ok = QtWidgets.QPushButton("OK")
        botoes.addWidget(self.botao_atualizar)
        botoes.addWidget(self.botao_ok)
        layout.addLayout(botoes)

        self.botao_atualizar.clicked.connect(self.plotar_variograma)
        self.botao_ok.clicked.connect(self.confirmar_parametros)

    def alternar_expoente(self, texto):
        self.exponent_input.setVisible(texto == "power")

    def calcular_variograma_empirico(self):
        dist = pdist(self.coords)
        diff = pdist(self.valores[:, None], metric='euclidean')
        semi = 0.5 * diff**2

        bins = np.linspace(0, np.max(dist), 15)
        indices = np.digitize(dist, bins)

        self.bin_centers = []
        self.semi_media = []
        for i in range(1, len(bins)):
            mask = indices == i
            if np.any(mask):
                self.bin_centers.append(np.mean(dist[mask]))
                self.semi_media.append(np.mean(semi[mask]))

        self.bin_centers = np.array(self.bin_centers)
        self.semi_media = np.array(self.semi_media)

    def definir_parametros_iniciais(self):
        nugget = min(self.semi_media)
        sill = max(self.semi_media)
        range_ = max(self.bin_centers) * 0.8

        self.nugget_input.setValue(nugget)
        self.sill_input.setValue(sill)
        self.range_input.setValue(range_)

    def modelo_variograma(self, h, nugget, sill, range_, modelo, exponent=1.95):
        h = np.array(h)
        if modelo == "spherical":
            return np.where(h <= range_,
                            nugget + (sill - nugget)*(1.5*(h/range_) - 0.5*(h/range_)**3),
                            sill)
        elif modelo == "exponential":
            return nugget + (sill - nugget)*(1 - np.exp(-h / (range_ / 3)))
        elif modelo == "gaussian":
            return nugget + (sill - nugget)*(1 - np.exp(-(h**2) / ((range_/3)**2)))
        elif modelo == "linear":
            return np.minimum(nugget + (sill - nugget)*(h / range_), sill)
        elif modelo == "power":
            return nugget + (sill - nugget)*(h / range_)**exponent
        else:
            return np.full_like(h, nugget)

    def plotar_variograma(self):
        nugget = self.nugget_input.value()
        sill = self.sill_input.value()
        range_ = self.range_input.value()
        modelo = self.modelo_input.currentText()
        exponent = self.exponent_input.value()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(self.bin_centers, self.semi_media, label="Empírico", color="blue")

        h_vals = np.linspace(0, max(self.bin_centers)*1.1, 100)
        gamma_vals = self.modelo_variograma(h_vals, nugget, sill, range_, modelo, exponent)

        ax.plot(h_vals, gamma_vals, label=f"Modelo: {modelo}", color="red")
        ax.set_xlabel("Distância (h)")
        ax.set_ylabel("Semivariância γ(h)")
        ax.legend()
        self.canvas.draw()

    def confirmar_parametros(self):
        nugget = self.nugget_input.value()
        sill = self.sill_input.value()
        range_ = self.range_input.value()
        modelo = self.modelo_input.currentText()
        exponent = self.exponent_input.value()
        self.parametros_confirmados.emit(nugget, sill, range_, modelo, exponent)
        self.accept()


def ajustar_variograma(coords, valores):
    widget = VariogramaWidget(coords, valores)
    resultado = {}

    def salvar_parametros(n, s, r, m, e):
        resultado["nugget"] = n
        resultado["sill"] = s
        resultado["range"] = r
        resultado["modelo"] = m
        resultado["exponent"] = e

    widget.parametros_confirmados.connect(salvar_parametros)

    if widget.exec_() == QtWidgets.QDialog.Accepted:
        return (resultado.get("nugget"),
                resultado.get("sill"),
                resultado.get("range"),
                resultado.get("modelo"),
                resultado.get("exponent"))
    else:
        return None, None, None, None, None
'''
