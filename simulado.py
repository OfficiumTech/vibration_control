# simulado.py

import os
import math
import numpy as np
import traceback
import time
from datetime import datetime
from qgis import processing
import tempfile
from osgeo import gdal, osr


from qgis.core import (
    QgsApplication,
    QgsColorRamp,
    QgsGradientColorRamp,
    QgsStyle,
    QgsSymbolLayerUtils,
    QgsPalettedRasterRenderer,
    QgsSingleBandPseudoColorRenderer,
    QgsColorRampShader,
    QgsRasterShader,
    edit,
    QgsPalLayerSettings,
    QgsProcessingFeedback,
    QgsRasterDataProvider,
    QgsRasterLayer,
    QgsGeometryUtils,
    QgsSpatialIndex,
    QgsDistanceArea,
    QgsMultiPoint,
    QgsVertexId,
    QgsLineString, 
    QgsMultiLineString,
    QgsRectangle,
    QgsVectorDataProvider,
    QgsCoordinateReferenceSystem,
    QgsFeatureSink,
    QgsVectorLayerSimpleLabeling,
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPoint,
    QgsPointXY,
    QgsVectorFileWriter,
    QgsField,
    QgsFields,
    QgsWkbTypes,
    QgsFeatureRequest
)

from qgis.PyQt.QtWidgets import QApplication
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from PyQt5.QtCore import QVariant, QCoreApplication
#from qgis.PyQt.QtCore import QVariant
#from qgis.core import QgsProject
#from PyQt5.QtCore import QCoreApplication  # Progress Bar
from .processamento_sim import *
from .variograma import ajustar_variograma
from .krigagem import krigagemF

def verificar_pykrige():
    try:
        from pykrige.ok import OrdinaryKriging

        # Cria uma janela popup perguntando ao usuário se deseja usar Krigagem
        msg_box = QMessageBox()
        msg_box.setWindowTitle("PyKrige disponível")
        msg_box.setText("A biblioteca PyKrige está disponível nesta instalação.\n'OK' para usar krigagem ou 'Cancel' para usar IDW")
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Ok)
        resposta = msg_box.exec_()

        if resposta == QMessageBox.Ok:
            krige_disponivel = True
        else:
            krige_disponivel = False

    except ImportError:
        krige_disponivel = False
    
    return krige_disponivel

def plotar_grade_xy_no_qgis(grid_x, grid_y, iface):
    """
    Cria e plota uma camada temporária de pontos no QGIS representando a grade (grid_x, grid_y).
    
    Argumentos:
    - grid_x, grid_y: matrizes numpy de coordenadas X e Y da grade (np.meshgrid).
    - iface: interface do QGIS.
    """
    # 1. Verifica dimensões
    n_rows, n_cols = grid_x.shape
    
    # 2. Cria camada de ponto temporária (geometry type e crs)
    crs_proj = QgsProject.instance().crs().authid()  # Ex: 'EPSG:31983'
    camada = QgsVectorLayer(f'Point?crs={crs_proj}', 'Grade XY', 'memory')
    prov = camada.dataProvider()
    
    # 3. Define campos (atributos)
    campos = QgsFields()
    campos.append(QgsField('id', QVariant.Int))
    campos.append(QgsField('x', QVariant.Double))
    campos.append(QgsField('y', QVariant.Double))
    prov.addAttributes(campos)
    camada.updateFields()
    
    # 4. Cria features para cada ponto da grade
    features = []
    contador = 0
    for i in range(n_rows):
        for j in range(n_cols):
            x = float(grid_x[i, j])
            y = float(grid_y[i, j])
            ponto = QgsPointXY(x, y)
            geom = QgsGeometry.fromPointXY(ponto)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feat.setAttributes([contador, x, y])
            features.append(feat)
            contador += 1

    # 5. Adiciona à camada e ao QGIS
    prov.addFeatures(features)
    camada.updateExtents()
    QgsProject.instance().addMapLayer(camada)
    log(f"Grade plotada, {contador} pontos adicionados.")

def plotar_pontos(x, y, z, v):
    # Cria camada de ponto Z temporária
    crs = QgsProject.instance().crs().authid()
    vl = QgsVectorLayer(f"Point?crs={crs}", "PPV_pontos", "memory")
    #vl = QgsVectorLayer("PointZ?crs=EPSG:31983", "PPV_pontos", "memory")
    pr = vl.dataProvider()
    
    # Adiciona campo PPV
    pr.addAttributes([QgsField("PPV", QVariant.Double)])
    vl.updateFields()

    # Cria e adiciona as feições
    feats = []
    for xi, yi, zi, vi in zip(x, y, z, v):
        pt = QgsPoint(xi, yi, zi)
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPoint(pt))
        feat.setAttributes([vi])
        feats.append(feat)

    pr.addFeatures(feats)
    vl.updateExtents()
    QgsProject.instance().addMapLayer(vl)

def log(mensagem):
    QMessageBox.information(None, "Simulado - Blaster Vibration Control", mensagem)

def obter_camada_por_nome(nome):
    for camada in QgsProject.instance().mapLayers().values():
        if camada.name() == nome:
            return camada
    return None

def obter_campo_z(layer):
    # Se a geometria for 2D, busca campo z na tabela de atributos
    campos_possiveis = ["z", "Z", "Cota", "Elevação", "Cota Z", "Cota z"]
    nomes_colunas = [field.name() for field in layer.fields()]
    for campo in campos_possiveis:
        if campo in nomes_colunas:
            log(f"Geometria 2D: campo de cota encontrado: '{campo}'")
            return campo

    # Se nenhum campo for encontrado, pergunta ao usuário
    campo, ok = QInputDialog.getItem(None,
                                      "Campo de Cota Z",
                                      "Geometria 2D detectada.\nEscolha o campo que representa a cota Z ou selecione 'Sem cota z':",
                                      nomes_colunas + ["Sem cota z"], editable=False)
    if ok:
        return campo if campo != "Sem cota z" else None
    return None

def distancia_3d(xf, yf, zf, xi, yi, zi):
    return math.sqrt((xf - xi) ** 2 + (yf - yi) ** 2 + (zf - zi) ** 2)

def distancia_3d_quadrado(xf, yf, zf, xi, yi, zi):
    return ((xf - xi) ** 2 + (yf - yi) ** 2 + (zf - zi) ** 2)
       
def gerar_malha(area_poligono, resolucao=20):
    # Cria uma malha de pontos dentro do polígono com espaçamento (em metros)
    #extent = area_poligono.extent()
    extent = area_poligono.boundingBox()
    xmin, xmax = extent.xMinimum(), extent.xMaximum()
    ymin, ymax = extent.yMinimum(), extent.yMaximum()
    
    pontos = []
    x = xmin
    while x <= xmax:
        y = ymin
        while y <= ymax:
            ponto = QgsPointXY(x, y)
            if area_poligono.contains(QgsGeometry.fromPointXY(ponto)):
                pontos.append(ponto)
            y += resolucao
        x += resolucao
    return pontos

def interpolar_ppv(pontos, valores, grid_x, grid_y, grid_z, uii):
    if krige_disponivel and grid_z is not None:
        try:
            from pykrige.ok3d import OrdinaryKriging3D

            ok3d = OrdinaryKriging3D(
                [p[0] for p in pontos],  # x
                [p[1] for p in pontos],  # y
                [p[2] for p in pontos],  # z
                valores,
                variogram_model='linear'
            )

            z_interp, _ = ok3d.execute('grid', grid_x, grid_y, grid_z)
            return z_interp  # shape: (len(grid_z), len(grid_y), len(grid_x))

        except Exception as e:
            log(f"Erro ao usar PyKrige 3D: {str(e)}. Usando IDW 3D.")
    
    else: # Fallback: IDW 3D otimizado       
        total_x = len(grid_x)
        total_y = len(grid_y)
        total_z = len(grid_z)
        total_pontos = total_x * total_y * total_z
        contador = 0

        z_interp = np.zeros((total_z, total_y, total_x))

        array_pontos = np.array(pontos)  # shape (N, 3)
        array_valores = np.array(valores)  # shape (N,)

        for k, z_val in enumerate(grid_z):
            for i, y_val in enumerate(grid_y):
                for j, x_val in enumerate(grid_x):
                    dx = array_pontos[:, 0] - x_val
                    dy = array_pontos[:, 1] - y_val
                    dz = array_pontos[:, 2] - z_val
                    dist_sq = dx**2 + dy**2 + dz**2

                    if np.any(dist_sq == 0):
                        valor = array_valores[dist_sq == 0][0]
                    else:
                        pesos = 1 / dist_sq
                        valor = np.sum(pesos * array_valores) / np.sum(pesos)

                    z_interp[k, i, j] = valor

                    # Atualiza progress bar a cada 1%
                    contador += 1
                    if contador % max((total_pontos // 100), 1) == 0:
                        progresso = int((contador / total_pontos) * 100)
                        uii.progressBar_forPonto.setValue(min(progresso, 100))
                        QCoreApplication.processEvents()

        uii.progressBar_forPonto.setValue(100)
        return z_interp  # shape: (len(grid_z), len(grid_y), len(grid_x))

def obter_z_do_furo_mais_proximo(xi, yi, resultados_processados, log_func):
    """
    Retorna a cota Z do furo mais próximo ao ponto (xi, yi).
    Se a cota do furo mais próximo for inválida, retorna a média das cotas válidas.
    Parâmetros:
        xi, yi: Coordenadas do ponto simulado
        resultados_processados: Lista de dicionários com dados dos furos
        log_func: Função para registrar mensagens (ex: log)
    Retorna:
        zi: valor da cota Z estimada
        None: se nenhum furo tiver cota válida
    """
    # Lista com cotas Z válidas
    z_furos_validos = []
    for dado in resultados_processados:
        zf = dado.get('z do furo sim')
        if zf is not None and not (isinstance(zf, float) and np.isnan(zf)):
            z_furos_validos.append(zf)

    # Verifica se há dados válidos
    if not z_furos_validos:
        log_func("❌ Nenhum furo possui cota Z válida. Corrija a camada de furos.")
        return None

    # Calcula a média como fallback
    media_z_furos = sum(z_furos_validos) / len(z_furos_validos)

    # Busca o furo mais próximo
    menor_dist = float('inf')
    zi = media_z_furos  # valor padrão caso o mais próximo tenha Z inválido

    for dado in resultados_processados:
        xf, yf, zf = dado.get('x do furo sim'), dado.get('y do furo sim'), dado.get('z do furo sim')
        dist = ((xf - xi) ** 2 + (yf - yi) ** 2)

        if dist < menor_dist:
            menor_dist = dist
            zi = zf if zf is not None and not (isinstance(zf, float) and np.isnan(zf)) else media_z_furos

    return zi

def medir_tempo(func):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        print(f"[{func.__name__}] Tempo de execução: {fim - inicio:.2f} segundos")
        return resultado
    return wrapper
   
def dentro_triangulo(dicionario_vertices, xi, yi, ids):
    pSim_x = xi
    pSim_y = yi

    coords = []

    # Coleta de coordenadas em uma lista
    for fid in ids:
        feat = dicionario_vertices.get(fid)
        if not feat:
            continue
        for ponto in feat.geometry().vertices():
            coords.append((ponto.x(), ponto.y(), ponto.z()))

    if not coords:
        return None

    # Transforma em array NumPy
    coords_np = np.array(coords)  # shape: (N, 3)
    dx = coords_np[:, 0] - pSim_x
    dy = coords_np[:, 1] - pSim_y

    # Azimute em graus [0, 360)
    deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

    # Agora aplicamos os setores rotacionados
    for inc in (0, 45, 75):
        pontos_N = coords_np[(deg >= 0 + inc) & (deg <= 45 + inc)]
        pontos_SL = coords_np[(deg >= 120 + inc) & (deg <= 165 + inc)]
        pontos_SO = coords_np[(deg >= 240 + inc) & (deg <= 285 + inc)]

        if len(pontos_N) > 0 and len(pontos_SL) > 0 and len(pontos_SO) > 0:
            return pontos_N.tolist(), pontos_SL.tolist(), pontos_SO.tolist()

    return None

def arredonda_campo(layer: QgsVectorLayer, nome_campo: str):
    """Arredonda os valores do campo especificado e aplica rótulos à camada."""
    provider = layer.dataProvider()
    idx = layer.fields().indexFromName(nome_campo)

    if idx == -1:
        log(f"⚠️ Campo '{nome_campo}' não encontrado para arredondamento. Arredondamento ignorado")
        return

    if not provider.capabilities() & QgsVectorDataProvider.ChangeAttributeValues:
        log("⚠️ Camada não permite edição de atributos. Arredondamento ignorado.")
        return

    # Arredonda os valores do campo
    with edit(layer):
        for feat in layer.getFeatures():
            valor = feat[nome_campo]
            try:
                if valor is not None:
                    valor_arredondado = round(float(valor), 2)
                    provider.changeAttributeValues({feat.id(): {idx: valor_arredondado}})
            except Exception as e:
                log(f"⚠️ Erro ao arredondar valor '{valor}': {e}")

    # Aplica rótulo ao campo
    if layer and layer.isValid():
        labeling = QgsPalLayerSettings()
        labeling.fieldName = nome_campo
        labeling.placement = QgsPalLayerSettings.Curved
        labeling.enabled = True

        label_settings = QgsVectorLayerSimpleLabeling(labeling)
        layer.setLabeling(label_settings)
        layer.setLabelsEnabled(True)
        layer.triggerRepaint()

# Aplica simbologia a camada raster
def conf_raster(layer, valores_ppv, Output_Layer_Name):
    if not isinstance(layer, QgsRasterLayer):
        log("A camada fornecida não é um raster.")
        return

    # Obter estatísticas da banda 1
    stats = layer.dataProvider().bandStatistics(1)
    minVal = stats.minimumValue
    maxVal = stats.maximumValue

    if minVal == maxVal:
        log("Valores mínimos e máximos são iguais. Não é possível aplicar rampa de cor.")
        return

    # Obter estilo de cor
    #qgis_version = QgsApplication.qgisVersion()
    style = QgsStyle().defaultStyle()
    ramp_name = 'RdYlGn'

    if not style.colorRampNames().__contains__(ramp_name):
        log(f"Rampa de cor '{ramp_name}' não encontrada no estilo do QGIS.")
        return

    ramp = style.colorRamp(ramp_name)

    # Criar renderer===================================================== Novo
    color_shader = QgsColorRampShader() # Define um shader com valores únicos (discretos)
    color_shader.setColorRampType(QgsColorRampShader.Discrete)  # ou Interpolated, se preferir

    # Gera uma rampa 'Spectral' automaticamente com os valores únicos
    entries = []
    valores_unicos = sorted(set(valores_ppv))
    ramp = QgsStyle().defaultStyle().colorRamp('Spectral')  # usa a rampa 'Spectral' do QGIS

    # Cria entradas discretas associando cada valor a uma cor da rampa
    for i, val in enumerate(valores_unicos):
        cor = ramp.color(float(i) / max(1, len(valores_unicos)-1))
        entries.append(QgsColorRampShader.ColorRampItem(val, cor, str(round(val, 2))))

    color_shader.setColorRampItemList(entries)

    # Aplica o renderizador com o shader
    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(color_shader)    # Envolver no shader raster
    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, raster_shader
    )
    layer.setRenderer(renderer)
    layer.setName(Output_Layer_Name)
    layer.triggerRepaint()

#Faz a interpolação
def gerar_curvas_isovalores(pontos_ppv, valores_ppv, camada_base, log, uii, resolucao):
    """
    Gera curvas de isovalores (contornos) de PPV a partir de pontos de vibração e valores,
    utilizando interpolação 3D (Kriging ou IDW), gera raster temporário e curvas com GDAL.

    Argumentos:
    - pontos_ppv: lista de coordenadas 3D [(x, y, z), ...]
    - valores_ppv: lista de valores PPV correspondentes
    - camada_base: camada de referência para CRS
    - nome_raster_saida: nome do arquivo raster a ser salvo
    - nome_curvas_saida: nome do arquivo vetorial das curvas
    - log: função para registrar mensagens no UI
    - uii: objeto da interface com barra de progresso
    """
    
    #Cota Z
    def cota(tolerancia, x, y, z, x0, y0):
        raioBusca = tolerancia
        dists = np.sqrt((x - x0)**2 + (y - y0)**2)
        minVizinho = 0

        while minVizinho <= 5:
            mask = dists <= raioBusca
            minVizinho = np.sum(mask)

            raioBusca += 20
            if raioBusca >= 500:
                break

        if raioBusca > 500 and minVizinho <= 5:
            return None

        # Calcula a interpolação
        dists_vizinhos = dists[mask]
        z_vizinhos = z[mask]

        if np.any(dists_vizinhos == 0):
            zinterpolado = np.mean(z_vizinhos[dists_vizinhos == 0])
        else:
            pesos = 1 / dists_vizinhos**2
            zinterpolado = np.sum(pesos * z_vizinhos) / np.sum(pesos)
            
        return zinterpolado
    #Fim Cota============================
    
    
    
    try:
        # Converte dados calculados de entrada em arrays numpy para facilitar a manipulação
        x = np.array([p[0] for p in pontos_ppv])
        y = np.array([p[1] for p in pontos_ppv])
        z = np.array([p[2] for p in pontos_ppv])
        v = np.array(valores_ppv)
        
        # Adiciona ao projeto para depuração
        #plotar_pontos(x, y, z, v)

        # Define resolução e cria a grade regular a partir das coordenadas calculadas 'pontos_ppv'
        grid_res = 10  # tamanho de célula em metros (10x10m)
        xi = np.arange(x.min(), x.max(), grid_res)
        yi = np.arange(y.min(), y.max(), grid_res)
        zi = np.arange(z.min(), z.max(), grid_res) 
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        #plotar_grade_xy_no_qgis(grid_x, grid_y, uii) #Verificar a malha
        
        krige_disponivel = verificar_pykrige() #Verifica se a biblioteca pykrige está presente e se o usuário deseja usá-la
        
        
        #=== Interpolação por PyKrige  ===========================================================#
        #=========================================================================================#
        
        try:
            # Interpolação com PyKrige (se disponível)
            if krige_disponivel:
                from pykrige.ok import OrdinaryKriging
                
                # Chama a interface de ajuste de semi-variograma-----------------------------------
                coords = list(zip(x, y, z))
                valores = v
                           
                nugget, sill, range_, modelo, exponent, scale = ajustar_variograma(coords, valores)
                if None in (nugget, sill, range_, modelo):
                    log("⚠️ Ajuste do variograma cancelado. Usando IDW como fallback.")
                    raise ValueError("Ajuste cancelado.")
  
                log(f"📊 Parâmetros ajustados: modelo={modelo}, nugget={nugget:.3f}, sill={sill:.3f}, range={range_:.2f}")
                # Fim ajuste-----------------------------------------------------------------------
                
                # Chama a função que faz Krigagem--------------------------------------------------
                #include_z_as_var=interpolaZ,
                interpolaZ = False
                grid_result = krigagemF(nugget=nugget, sill=sill, range_=range_, 
                    modelo=modelo, exponent=exponent, scale=scale, x=x, y=y, z=z, v=v, 
                    grid_x=grid_x, grid_y=grid_y, uii=uii, tolerancia=100, log=log
                    )
                
                # Verificar  NaN
                if np.isnan(grid_result).sum() > (grid_result.size / 2):
                    log("⚠️ Mais da metade do resultado da krigagem contém NaN. Usando IDW como fallback.")
                    raise ValueError("Resultado da krigagem inválido.")

                log("✅ Interpolação por Krigagem concluída.")
                #-----------------------------------------------------------------------------------
                
            else:
                raise ImportError("Biblioteca PyKrige não disponível")

        
        #=== Interpolação por IDW  ===========================================================#
        #=====================================================================================#
        
        except Exception as e:
            tb = traceback.format_exc()
            # Fallback: Interpolação IDW 3D manual
            log(f"⚠️ Interpolando por IDW 3D como fallback. \nERRO: \n{e} \nTraceback:\n{tb}")
            grid_result = np.zeros_like(grid_x, dtype=float)

            # Empilha coordenadas em um único array (Nx3)
            coords_interp = np.column_stack((x, y, z))
            num_pontos = coords_interp.shape[0]
            total_celulas = grid_x.size # Total de pontos da grade (para barra de progresso)
            celulas_processadas = 0

            distPegaCota = 20
            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    px = grid_x[i, j] # Coordenada do ponto da grade
                    py = grid_y[i, j]
                    pz = cota(distPegaCota, x, y, z, px, py)
                    
                    if pz == None or np.isnan(pz):
                        continue

                    # Calcula distâncias 3D entre o ponto da grade e os pontos conhecidos
                    distancias = np.sqrt((coords_interp[:, 0] - px)**2 +
                                         (coords_interp[:, 1] - py)**2 +
                                         (coords_interp[:, 2] - pz)**2)

                    # Seleciona os 5 vizinhos mais próximos
                    k = min(5, num_pontos)
                    idx = np.argsort(distancias)[:k]
                    dist_vizinhos = distancias[idx]

                    # Interpola com pesos inversamente proporcionais à distância
                    if np.any(dist_vizinhos == 0):
                        grid_result[i, j] = np.mean(v[idx][dist_vizinhos == 0])
                    else:
                        pesos = 1 / dist_vizinhos**2
                        grid_result[i, j] = round(np.sum(pesos * v[idx]) / np.sum(pesos), 3)

                    # Atualiza barra de progresso a cada 1% concluído
                    celulas_processadas += 1
                    if total_celulas > 0 and celulas_processadas % max(total_celulas // 100, 1) == 0:
                        progresso = int((celulas_processadas / total_celulas) * 100)
                        uii.progressBar_forPonto.setValue(min(progresso, 99))  # deixa 100% para final
                        QCoreApplication.processEvents()



        #=== Exporta grade interpolada como raster  ==========================================#
        #=====================================================================================#
        def gera_raster_2e3D(grid_result, grid_res, xi, yi, camada_base, uii, valores_ppv=None, log=print):
            # Detecta dimensões do grid_result
            if grid_result.ndim == 2:
                nrows, ncols = grid_result.shape
                nbands = 1
            elif grid_result.ndim == 3:
                nrows, ncols, nbands = grid_result.shape
            else:
                raise ValueError("grid_result deve ser 2D (nx,ny) ou 3D (nx,ny,n_bandas)")

            # Extremos / resolução (preservo suas variáveis xi, yi, grid_res)
            x_min, x_max = xi.min(), xi.max()
            y_min, y_max = yi.min(), yi.max()
            pixel_size = grid_res  # Tamanho do pixel em metros

            # Nome do arquivo
            agora = datetime.now()
            timestamp = agora.strftime("%Y%m%d_%H%M%S")
            nome_raster_saida = f"raster_ppv_{timestamp}.tif"
            raster_path = os.path.join(tempfile.gettempdir(), nome_raster_saida)

            # Cria GeoTIFF com número de bandas adequado
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(raster_path, ncols, nrows, nbands, gdal.GDT_Float32)

            # Definir nodata e escrever bandas
            nodata = -9999.0

            if nbands == 1:
                arr = np.array(grid_result, dtype=np.float32)
                # substituir NaN por nodata e inverter eixo Y
                arr_write = np.where(np.isfinite(arr), arr, nodata)[::-1, :]
                band = dataset.GetRasterBand(1)
                band.WriteArray(arr_write)
                band.SetNoDataValue(nodata)
            else:
                # multibanda: escreve cada banda
                for b in range(nbands):
                    arr = np.array(grid_result[:, :, b], dtype=np.float32)
                    arr_write = np.where(np.isfinite(arr), arr, nodata)[::-1, :]
                    band = dataset.GetRasterBand(b + 1)
                    band.WriteArray(arr_write)
                    band.SetNoDataValue(nodata)

            # Define a transformação geoespacial (origem canto superior esquerdo)
            # nota: driver.Create usa (xsize=ncols, ysize=nrows)
            dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

            # Define sistema de referência com base na camada base
            srs = osr.SpatialReference()
            srs.ImportFromWkt(camada_base.crs().toWkt())
            dataset.SetProjection(srs.ExportToWkt())

            # Finaliza e salva
            dataset.FlushCache()
            dataset = None

            # Carrega no QGIS
            raster_layer = QgsRasterLayer(raster_path, "Raster PPV")
            if raster_layer.isValid():
                QgsProject.instance().addMapLayer(raster_layer)
                # tenta aplicar simbologia só se conf_raster existir e for chamável
                try:
                    # se existir variável valores_ppv no escopo, usa-a; senão, chama sem ela
                    if 'valores_ppv' in globals() or 'valores_ppv' in locals():
                        conf_raster(raster_layer, valores_ppv, "Raster PPV_R")
                    else:
                        # se for multibanda, você talvez queira aplicar simbologia na banda 1
                        conf_raster(raster_layer, None, "Raster PPV_R")
                except Exception:
                    # se conf_raster não existir ou falhar, apenas ignore
                    pass

            uii.progressBar_forPonto.setValue(100)
            log(f"✅ Raster de PPV salvo em: {raster_path}")
            
            return raster_layer
    
        def gera_raster(grid_result, grid_res, xi, yi, camada_base, uii, valores_ppv=None, log=print):
            # Dimensões do grid interpolado ------------------------------------------------------
            nrows, ncols = grid_result.shape
            x_min, x_max = xi.min(), xi.max()
            y_min, y_max = yi.min(), yi.max()
            pixel_size = grid_res # Tamanho do pixel em metros


            #Criar nome raster único com data e hora ---------------------------------------------
            agora = datetime.now()
            timestamp = agora.strftime("%Y%m%d_%H%M%S")  # 20250505_153022
            nome_raster_saida = f"raster_ppv_{timestamp}.tif"
            raster_path = os.path.join(tempfile.gettempdir(), nome_raster_saida)    # Caminho temporário onde o raster será salvo

            # Criação do GeoTIFF com GDAL --------------------------------------------------------
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(raster_path, ncols, nrows, 1, gdal.GDT_Float32) # O '1' é o nmr de bandas
            dataset.GetRasterBand(1).WriteArray(grid_result[::-1])  # '[::-1]' inverte eixo Y (imagem)

            # Define a transformação geoespacial: origem no canto superior esquerdo e resolução---
            # O valor negativo no 'pixel_size' indica que os pixels crescem de cima para baixo, como esperado pelo QGIS
            dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

            # Define o sistema de coordenadas do raster com base na camada base-------------------
            srs = osr.SpatialReference()
            srs.ImportFromWkt(camada_base.crs().toWkt())
            dataset.SetProjection(srs.ExportToWkt())

            # Finaliza e salva o raster ----------------------------------------------------------
            dataset.FlushCache()
            dataset = None

            # Carrega raster no QGIS -------------------------------------------------------------
            raster_layer = QgsRasterLayer(raster_path, "Raster PPV")
            if raster_layer.isValid():
                #valores_ppv = grid_result.flatten().tolist()
                QgsProject.instance().addMapLayer(raster_layer)
                conf_raster(raster_layer, valores_ppv, "Raster PPV_R") # Aplica simbologia ao raster
            uii.progressBar_forPonto.setValue(100)  # Conclui barra de progresso
            log(f"✅ Raster de PPV salvo em: {raster_path}")
        
        raster_layer = gera_raster_2e3D(grid_result, grid_res, xi, yi, camada_base, uii, valores_ppv, log)



        #=== Geração de curvas de nível (contorno) ===========================================#
        #=====================================================================================#
        
        v = np.array([float(val) for val in valores_ppv])
        
        intervalo = float(0.5)  # valor padrão
        try:
            intervaloGui = uii.lineEdit_iso.text()
            #log(f"IntervaloGui = {intervaloGui}")
            intervalo = float(intervaloGui)
            if intervalo < 0.1:
                intervalo = float(0.1)
            #log(f"Intervalo PPV = {intervalo} mm/s")
        except: #
            #log(f"Campo de espaçamento entre linhas vazio. Usando valor padrão ({intervalo} mm/s).")
            intervalo = float(0.5)

        # Criar camada única com data e hora
        agora = datetime.now()
        timestamp = agora.strftime("%Y%m%d_%H%M%S")  # 20250505_153022
        nome_curvas_saida = f"curvas_ppv_{timestamp}.shp"
        
        # Caminho de saída para as curvas de isovalores
        output_curvas = os.path.join(tempfile.gettempdir(), nome_curvas_saida)
        
        # Gera as curvas com o algoritmo GDAL "Contour"
        processing.run("gdal:contour", {
            'INPUT':raster_layer,
            'BAND':1,
            'INTERVAL':intervalo,
            'FIELD_NAME':'PPV',
            'CREATE_3D':True,
            'IGNORE_NODATA':True,
            'NODATA':-9999,
            'OFFSET':0,
            'EXTRA':'',
            'OUTPUT':output_curvas
        }, feedback=QgsProcessingFeedback())

        
        # === Suaviza as curvas usando o algoritmo "Smooth Geometry" do QGIS ====================
        
        # Caminho de saída para as curvas suavizadas
        nome_curvas_suavizadas = f"curvas_ppv_suave_{timestamp}.shp"
        output_curvasS = os.path.join(tempfile.gettempdir(), nome_curvas_suavizadas)
        
        processing.run("qgis:smoothgeometry", {
            'INPUT': output_curvas,
            'ITERATIONS': 2,  # número de iterações (2–5 costuma ser suficiente)
            'OFFSET': 0.25,   # fator de suavização (ajuste conforme necessário)
            'MAX_ANGLE': 180, # ângulo máximo entre segmentos (opcional)
            'OUTPUT': output_curvasS
        }, feedback=QgsProcessingFeedback())

        log(f"✅ Curvas de isovalores salvas em: {output_curvasS}")
        
        
        # ========Adiciona as curvas ao projeto==========================================
        curvas_layer = QgsVectorLayer(output_curvasS, "Curvas PPV", "ogr")   
        if not curvas_layer.isValid():
            log("⚠️ Erro ao carregar camada de curvas.")
        else:
            curvas_layer.setSubsetString('"PPV" <= 20') # Corta isolinhas com valores abaixo de 10
            QgsProject.instance().addMapLayer(curvas_layer)
        #-----------------------------------------------------------------------
        # Força carregamento dos campos para garantir acesso ao campo 'PPV'
        curvas_layer.dataProvider().reloadData()

        # Arredonda e aplica os valores do campo 'PPV'--------------------------
        arredonda_campo(curvas_layer, 'PPV')
        
    except Exception as e: 
        tb = traceback.format_exc()                        
        log(f"⚠️ Erro na curva de isovalores: {e} \nTraceback:\n{tb}")

# Cria o indice espacial da camada de topografia
def indice_espacial_topo(ui, geometria_area, camada_topo, camada_topo3D, campo_z, log):
    if camada_topo:
        contadorBar = 0
        feats_total = camada_topo.featureCount()
        
        # Criar camada temporária de pontos
        uri = "PointZ?crs=" + camada_topo.crs().authid()
        layer_vertices = QgsVectorLayer(uri, "vertices_topo", "memory")
        provider = layer_vertices.dataProvider()
        campos = QgsFields()
        campos.append(QgsField("id_original", QVariant.Int))
        provider.addAttributes(campos)
        layer_vertices.updateFields()

        feats_vertices = []

        # Índice espacial para verificação rápida de pontos já adicionados
        pontos_adicionados_index = QgsSpatialIndex()
        pontos_adicionados = {}  # dicionário para armazenar QgsFeature temporárias

        repetir = 0
        dist_p = 4.99
        #geometria_areaPlus = geometria_area.buffer(50, 1)
        for feat in camada_topo.getFeatures():
            # ProgressBar
            contadorBar += 1
            if feats_total > 0 and contadorBar % max((feats_total // 100), 1) == 0:
                progresso = int((contadorBar / feats_total) * 100)
                ui.progressBar_forPonto.setValue(min(progresso, 98))
                QCoreApplication.processEvents()
            
            geom = feat.geometry()
            '''or not geom.intersects(geometria_areaPlus):'''
            if not geom:
                continue
                
            for point in geom.vertices():
                ponto_geom = QgsGeometry.fromPoint(point)
                '''if not ponto_geom.intersects(geometria_areaPlus):
                    continue'''
                repetir += 1
                try:
                    x, y = point.x(), point.y()
                    if camada_topo3D == True:
                        z = point.z()
                    else:
                        if campo_z:
                            z = float(feat[campo_z]) #Se a camada for 2D e tiver campo Z na tab. de atributos
                        else:
                            z = obter_z_do_furo_mais_proximo(x, y, ui.resultados_processadosFsim, log) # Puxa o z da boca do furo de simulação. Só p/ testes
                        if np.isnan(z):
                            continue
                    
                    # Cria geometria do novo ponto
                    novo_ponto_geom = QgsGeometry.fromPoint(QgsPoint(x, y, z))

                    # Verifica se já existe ponto próximo usando índice espacial
                    bbox = novo_ponto_geom.boundingBox()
                    bbox.grow(dist_p)  # cresce 4.99 metro em volta
                    ids_proximos = pontos_adicionados_index.intersects(bbox)

                    ponto_proximo = False
                    for id_candidato in ids_proximos:
                        candidato_geom = pontos_adicionados[id_candidato]
                        if candidato_geom.distance(novo_ponto_geom) < dist_p: #Pontos a menos de 5m de dist. são descartados
                            ponto_proximo = True
                            break

                    if ponto_proximo:
                        continue  # pula se encontrou ponto muito próximo

                    # Se chegou aqui, é um novo ponto válido
                    feat_nova = QgsFeature(layer_vertices.fields())
                    ponto_z = QgsPoint(x, y, z)
                    feat_nova.setGeometry(QgsGeometry.fromPoint(ponto_z))
                    feat_nova.setAttribute("id_original", feat.id())
                    feats_vertices.append(feat_nova)

                    # Atualiza índice espacial
                    id_temporario = len(pontos_adicionados)
                    pontos_adicionados[id_temporario] = feat_nova.geometry()
                    pontos_adicionados_index.addFeature(feat_nova)

                except Exception as e:
                    continue

        provider.addFeatures(feats_vertices)
        layer_vertices.updateExtents()

        # Adiciona ao projeto para depuração
        #QgsProject.instance().addMapLayer(layer_vertices)

        # Agora cria o índice espacial final com os vértices
        indice_topo = QgsSpatialIndex(layer_vertices.getFeatures())
        
        #Dicionário do layer_vertices para buscas: feature.id() ➔ chave / feature ➔ valor
        dicionario_vertices = {feature.id(): feature for feature in layer_vertices.getFeatures()}


        # ProgressBar
        ui.progressBar_forPonto.setValue(100)

        # Verificação final de processo
        num_vertices = layer_vertices.featureCount()
        if num_vertices == 0:
            log("⚠️ Nenhum vértice foi criado a partir da camada de topografia.")
        else:
            log(f"✅ Gerado índice espacial da camada de topografia, vértices criados: {num_vertices}")
        
        
        return layer_vertices, indice_topo, dicionario_vertices


#------------------------------------------------------------------------------------------------------------
#----------Função Principal----------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def executar_simulacao(ui, iface):
    try:
        ui.progressBar_forPonto.setVisible(True)
        ui.progressBar_forPonto.setValue(0)
        
        #Executa o processamento do plano de fogo de simulação
        processar_furoSim(ui)
        
        if not hasattr(ui, "resultados_processados") or not ui.resultados_processados:
            log("Nenhum dado processado encontrado. Execute a etapa de processamento antes.")
            return
        
        if not hasattr(ui, "resultados_processadosFsim") or not ui.resultados_processadosFsim:
            log("Adicione camada de plano de fogo de simulação antes em: 'Posição dos Furos de Desmonte a Simular' na aba INPUT")
            return
        
        if not hasattr(ui, "parametros_regressao"):
            log("Parâmetros de regressão não encontrados. Primeiro Gere a Lei de Atenuação")
            return

        a, b, c = ui.parametros_regressao
        #vib_max = float(ui.textEdit_VibraMax.toPlainText()) if ui.textEdit_VibraMax.toPlainText() else 0
        #carga_substituta = float(ui.textEdit_Carga.toPlainText()) if ui.textEdit_Carga.toPlainText() else 0

        #Obtenção da zona de simulação e da topografia
        nome_zcriticas = ui.comboBox_shpZCriticas.currentText()
        nome_topo = ui.comboBox_shpTopo.currentText()
        camada_zcriticas = obter_camada_por_nome(nome_zcriticas)
        camada_topo = obter_camada_por_nome(nome_topo)
        if not camada_zcriticas or not camada_topo:
            log("Selecione as camadas de área de simulação e topografia.")
            return
        if camada_topo.crs() != camada_zcriticas.crs():
            log("Atenção as camadas de área de simulação e topografia devem ter o mesmo CRS.")
        
        # verifica se a camada de topogrfia possui coordenadas Z na geometria
        camada_topo3D = False       
        if QgsWkbTypes.hasZ(camada_topo.wkbType()):
            camada_topo3D = True
            
        # Verifica se a camada de topografia é 2D e chama "obter_campo_z" p/ pegar as cotas na tab atrib.
        campo_z = None
        if camada_topo3D == False:
            campo_z = obter_campo_z(camada_topo)
            
        # Obtém geometria do primeiro polígono da camada de áreas críticas
        feat_area = next(camada_zcriticas.getFeatures(), None)
        if not feat_area:
            log("Não foi possível obter a geometria da área de simulação. Certifique-se de haver um único polígono na camada.")
            return
        geometria_area = feat_area.geometry()
        
        
        # Criar índice espacial-------------------------------------------------------------
        layer_vertices, indice_topo, dicionario_vertices = indice_espacial_topo(ui, geometria_area, camada_topo, camada_topo3D, campo_z, log)
        #-----------------------------------------------------------------------------------

        # Gera a malha de simulação
        resolucao = 20 #Resolução da malha de simulação Ex.: 20x20 metros
        malha = gerar_malha(geometria_area, resolucao)
        quantidade_pontos = len(malha)
        log(f"✅ Gerado malha quadrada para simulação, quantidade de pontos: {quantidade_pontos}")
        
        pontos_ppv = []
        valores_ppv = []
        
        # P/ Debug
        contErro = [0,0]
        vezesLog = 1
        ver = 0
        vezesVer = 2
        pri = None
        seg = None
        ter = None
        
        #Verificação de topografia
        pontos_descartados = 0
        
        # Reinicia o progressBar
        ui.progressBar_forPonto.setVisible(True)
        ui.progressBar_forPonto.setValue(0)
        contadorBar2 = 0
        
        
        #For malha
        for ponto in malha:
            contErro[0] += 1
            
            contadorBar2 += 1
            if quantidade_pontos > 0 and contadorBar2 % max((quantidade_pontos // 100), 1) == 0:
                progresso = int((contadorBar2 / quantidade_pontos) * 100)
                ui.progressBar_forPonto.setValue(min(progresso, 100))
                QCoreApplication.processEvents()  # mantém interface responsiva
            
            #Coordenadas da malha
            xi, yi = ponto.x(), ponto.y()
            zi = None   # Variável que armazenará o Z a ser buscado
            
            # Área de busca ao redor do ponto simulado
            tolerancia = 100 # Tolerancia p/ caixa de busca.
            toleMax = 500
            ids = []
            resultado = None
            while resultado is None:
                rect = QgsRectangle(
                    xi - tolerancia, yi - tolerancia, 
                    xi + tolerancia, yi + tolerancia
                )
                ids = indice_topo.intersects(rect)
                
                if (len(ids) >= 3):
                    resultado = dentro_triangulo(dicionario_vertices, xi, yi, ids)
                    if resultado is not None:
                        pontos_N, pontos_SL, pontos_SO = resultado
                        #log(f"Pontos encontrados! Norte: {pontos_N}, Sudeste: {pontos_SL}, Sudoeste: {pontos_SO}")
                        break
                        
                tolerancia += 200   
                
                if tolerancia > toleMax:
                    #if contErro[0] <= vezesLog:
                    #log("Topografia muito longe do ponto simulado, pulando ponto isolado...")
                    break
            
            #Se não há pontos de topografia próximo que gerem um triangulo ao redor de (xi,yi); pula p/ próximo (xi,yi)
            if tolerancia > toleMax:
                continue
                #Processamento dos pontos da malha
                #quantidade_pontos -= 1
                #print(f"Quantidade de pontos na malha simulada: {quantidade_pontos}")
  
            z = None  # Inicializa Z             
 
            # Triangulação: pegar os 3 mais próximos
            def dist_sq(p):
                return (p[0] - xi) ** 2 + (p[1] - yi) ** 2
            p1 = min(pontos_N, key=dist_sq)
            p2 = min(pontos_SL, key=dist_sq)
            p3 = min(pontos_SO, key=dist_sq)

            # Calcula área dos triângulos para interpolar (barycentric coordinates)
            def area(px, py, qx, qy, rx, ry):
                return abs((px*(qy-ry) + qx*(ry-py) + rx*(py-qy)) / 2.0)
            
            area_total = area(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

            if area_total == 0: #Não é um triângulo
                z = (p1[2] + p2[2] + p3[2]) / 3  # fallback: média simples
            else:
                a1 = area(xi, yi, p2[0], p2[1], p3[0], p3[1]) / area_total
                a2 = area(xi, yi, p1[0], p1[1], p3[0], p3[1]) / area_total
                a3 = area(xi, yi, p1[0], p1[1], p2[0], p2[1]) / area_total
                
                epsilon = 1e-6  # tolerância
                if (0 <= a1 <= 1) and (0 <= a2 <= 1) and (0 <= a3 <= 1) and (abs((a1 + a2 + a3) - 1) < epsilon):
                    z = a1 * p1[2] + a2 * p2[2] + a3 * p3[2]
                else: # Fora do triângulo
                    z = (p1[2] + p2[2] + p3[2]) / 3


            zi = z
            if zi is None or not np.isfinite(zi):
                log(f"⚠️ ATENÇÂO: Nenhuma coordenada Z válida foi encontrada na geometria da topografia próxima ao ponto simulado. \n Esse ponto será desconsiderado.")
                continue   

            try:
                # Encontra furo mais próximo - furo de simulação
                menor_dist = float('inf') #infinito positivo
                carga = 0
                for dado in ui.resultados_processadosFsim:
                    xf, yf, zf = dado['x do furo sim'], dado['y do furo sim'], dado['z do furo sim']
                    dist = distancia_3d_quadrado(xf, yf, zf, xi, yi, zi)
                    #log(f"xf: {xf} // yf: {yf} // zf: {zf} \n xi: {xi} // yi: {yi} // zi: {zi}")
                    if dist < menor_dist:
                        menor_dist = dist
                        carga = dado['Total Carga / Seq.']
                
                menor_dist_raiz = math.sqrt(menor_dist) # Valor de distancia
                #if carga_substituta > 0:
                    #carga = carga_substituta
                
                #Teste de resultado
                if menor_dist_raiz is None or not isinstance(menor_dist_raiz, (int, float)) or math.isnan(menor_dist_raiz):
                    continue
                # evita D = 0 e evita valores de vibração tendendo ao infinito
                if menor_dist_raiz < 0.01:
                    menor_dist_raiz = 0.01
                #Calculo de velocidade
                ppv = a * (carga ** b) * (menor_dist_raiz ** c)
                #Teste do resultado
                if ppv is None or not isinstance(ppv, (int, float)) or math.isnan(ppv):
                    continue
                #Atribuição
                pontos_ppv.append((xi, yi, zi))
                valores_ppv.append(ppv)
                
                #Debug
                #if contErro[0] <= vezesLog:
                    #log(f"Ponto simulado gerado: ({xi}, {yi}, {zi}), PPV = {ppv}")
                    #log(f"Final da função - Verificando coord. xi: {xi}, yi: {yi}, zi: {zi} ")

                    
            except Exception as e:     
                    tb = traceback.format_exc()                              
                    if contErro[0] <= vezesLog:
                        #\n🧪 WKT da geometria ID {feat.id()}: {geom.asWkt()}
                        log(f"⚠️ Erro: final da função: {e} \nTraceback:\n{tb}")
        #For malha fim
        

        # Progress Bar até esse ponto
        ui.progressBar_forPonto.setVisible(True)
        ui.progressBar_forPonto.setValue(100)
        if len(pontos_ppv) > 0:
            log(f"✅ Gerado malha de simulação 3D! Quantidade de pontos:  {len(pontos_ppv)}")
        else:
            log("⚠️ Falha ao fazer triangulação. Verificar camada de topografia.")
        
        
        '''# Adiciona ao projeto para depuração-------------------------------
        # cria camada temporária de pontos (memory layer)
        crs = QgsProject.instance().crs().authid()  # usa o CRS do projeto
        vl = QgsVectorLayer(f"Point?crs={crs}", "Pontos_PPV", "memory")

        prov = vl.dataProvider()

        # adiciona campo para armazenar valor de PPV
        prov.addAttributes([QgsField("PPV", QVariant.Double)])
        vl.updateFields()

        # cria feições a partir de pontos_ppv + valores_ppv
        feats = []
        for (x, y, z), ppv in zip(pontos_ppv, valores_ppv):
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            feat.setAttributes([ppv])
            feats.append(feat)

        prov.addFeatures(feats)
        vl.updateExtents()

        # adiciona camada ao projeto
        QgsProject.instance().addMapLayer(vl)
        #FIM---------------------------------------------------------------'''


        
        # Progress Bar até esse ponto
        ui.progressBar_forPonto.setVisible(True)
        ui.progressBar_forPonto.setValue(0)
        gerar_curvas_isovalores(
            pontos_ppv, valores_ppv,
            camada_topo, log,
            ui, resolucao  
        )
        

    except Exception as e:     
        tb = traceback.format_exc()                              
        #if contErro[0] <= vezesLog:
        log(f"⚠️ Erro na simulação: {e} \nTraceback:\n{tb}")
        #log(f"Erro na simulação: {str(e)}")


