# simulado.py

import os
import math
import numpy as np
import traceback
import time
from datetime import datetime
from qgis import processing


from qgis.core import (
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
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from PyQt5.QtCore import QVariant, QCoreApplication
#from qgis.PyQt.QtCore import QVariant
#from qgis.core import QgsProject
#from PyQt5.QtCore import QCoreApplication  # Progress Bar
from .processamento_sim import *


try:
    from pykrige.ok import OrdinaryKriging
    krige_disponivel = True
except ImportError:
    krige_disponivel = False


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


def salvar_shapefile(grid, xs, ys, crs_authid, zs=None, nome_arquivo="simulacao_ppv.shp"):
    """
    Salva os dados de PPV em um shapefile de pontos e adiciona ao projeto QGIS.
    """
    try:
        pasta_projeto = QgsProject.instance().homePath()
        caminho_completo = os.path.join(pasta_projeto, nome_arquivo)

        campos = QgsFields()
        campos.append(QgsField("PPV", QVariant.Double))

        is_3d = (zs is not None and len(grid.shape) == 3)
        if is_3d:
            campos.append(QgsField("Z", QVariant.Double))

        if not crs_authid:
            crs_authid = QgsProject.instance().crs().authid()

        crs = QgsCoordinateReferenceSystem(crs_authid)
        tipo_geom = QgsWkbTypes.PointZ if is_3d else QgsWkbTypes.Point

        writer = QgsVectorFileWriter(
            caminho_completo, "UTF-8", campos, tipo_geom, crs, "ESRI Shapefile"
        )

        if writer.hasError() != QgsVectorFileWriter.NoError:
            raise Exception(f"Erro ao criar o shapefile: {writer.errorMessage()}")

        # ✅ Contador de pontos salvos
        contador_pontos = 0

        if is_3d:
            for k, z in enumerate(zs):
                for i, y in enumerate(ys):
                    for j, x in enumerate(xs):
                        valor = grid[k][i][j]
                        if np.isnan(valor): #Debug
                            log(f"Valor NaN encontrado em ({x}, {y}, {z if zs else 'N/A'}): {valor}")
                        if not np.isnan(valor):
                            ponto = QgsPoint(x, y, z)
                            geometria = QgsGeometry.fromPoint(ponto)
                            feature = QgsFeature()
                            feature.setGeometry(geometria)
                            feature.setAttributes([float(valor), float(z)])
                            writer.addFeature(feature)
                            contador_pontos += 1  # incrementa contador
        else:
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    valor = grid[i][j]
                    if not np.isnan(valor):
                        ponto = QgsPointXY(x, y)
                        geometria = QgsGeometry.fromPointXY(ponto)
                        feature = QgsFeature()
                        feature.setGeometry(geometria)
                        feature.setAttributes([float(valor)])
                        writer.addFeature(feature)
                        contador_pontos += 1  # incrementa contador

        del writer

        log(f"✅ Shapefile salvo com {contador_pontos} pontos.")  # ✅ Log de pontos salvos

        camada = QgsVectorLayer(caminho_completo, nome_arquivo.replace(".shp", ""), "ogr")
        if not camada.isValid():
            raise Exception("Erro ao carregar o shapefile salvo no projeto.")

        QgsProject.instance().addMapLayer(camada)
        # 💡 Aplicar estilo leve para melhorar desempenho na visualização
        symbol = camada.renderer().symbol()
        symbol.setSize(0.2)  # Tamanho pequeno para muitos pontos
        symbol.setColor(QColor("blue"))  # Cor discreta
        camada.triggerRepaint()

    except Exception as e:
        log(f"❌ Erro ao salvar shapefile: {str(e)}")


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
        #log(f"✅ Campo '{nome_campo}' arredondado e rótulos aplicados.")
    
    
def gerar_curvas_isovalores(pontos_ppv, valores_ppv, camada_base, nome_raster_saida, nome_curvas_saida, log, uii):
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
    try:
        # Converte dados de entrada em arrays numpy para facilitar a manipulação
        x = np.array([p[0] for p in pontos_ppv])
        y = np.array([p[1] for p in pontos_ppv])
        z = np.array([p[2] for p in pontos_ppv])
        v = np.array(valores_ppv)

        # Define resolução e cria a grade regular 2D (XY)
        grid_res = 10  # tamanho de célula em metros (10x10m)
        xi = np.arange(x.min(), x.max(), grid_res)
        yi = np.arange(y.min(), y.max(), grid_res)
        zi = np.arange(z.min(), z.max(), grid_res)  # usado para Kriging 3D
        grid_x, grid_y = np.meshgrid(xi, yi)

        try:
            # Interpolação com PyKrige 3D (se disponível)
            if krige_disponivel:
                from pykrige.ok3d import OrdinaryKriging3D

                OK = OrdinaryKriging3D(x, y, z, v, variogram_model='linear')
                grid3d, ss3d = OK.execute('grid', xi, yi, zi)

                # Seleciona apenas a primeira fatia da interpolação em Z (nível fixo)
                grid_result = grid3d[:, :, 0]

                # Atualiza progressBar em 50% (considerando interpolação concluída)
                uii.progressBar_forPonto.setValue(80)
                log("✅ Interpolação com PyKrige 3D concluída.")
            else:
                raise ImportError("PyKrige não disponível")

        except Exception as e:
            # Fallback: Interpolação IDW 3D manual
            log(f"⚠️ Erro ao usar PyKrige 3D: {str(e)}. Usando IDW 3D como fallback.")
            grid_result = np.zeros_like(grid_x, dtype=float)

            # Empilha coordenadas em um único array (Nx3)
            coords = np.column_stack((x, y, z))
            num_pontos = coords.shape[0]

            # Total de pontos da grade (para barra de progresso)
            total_celulas = grid_x.shape[0] * grid_x.shape[1]
            celulas_processadas = 0

            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    # Coordenada do ponto da grade
                    px, py = grid_x[i, j], grid_y[i, j]
                    pz = np.mean(z)  # usar média de z (ou ajustar com topografia real)

                    # Calcula distâncias 3D entre o ponto da grade e os pontos conhecidos
                    distancias = np.sqrt((coords[:, 0] - px)**2 +
                                         (coords[:, 1] - py)**2 +
                                         (coords[:, 2] - pz)**2)

                    # Seleciona os 5 vizinhos mais próximos
                    k = min(5, num_pontos)
                    idx = np.argsort(distancias)[:k]
                    dist_vizinhos = distancias[idx]

                    # Interpola com pesos inversamente proporcionais à distância
                    if np.any(dist_vizinhos == 0):
                        grid_result[i, j] = np.mean(v[idx][dist_vizinhos == 0])
                    else:
                        pesos = 1 / dist_vizinhos
                        grid_result[i, j] = round(np.sum(pesos * v[idx]) / np.sum(pesos), 3)

                    # Atualiza barra de progresso a cada 1% concluído
                    celulas_processadas += 1
                    if total_celulas > 0 and celulas_processadas % max(total_celulas // 100, 1) == 0:
                        progresso = int((celulas_processadas / total_celulas) * 100)
                        uii.progressBar_forPonto.setValue(min(progresso, 99))  # deixa 100% para final
                        QCoreApplication.processEvents()

        # === Exporta grade interpolada como raster temporário ==========================================
        import tempfile
        from osgeo import gdal, osr

        # Dimensões do grid interpolado
        nrows, ncols = grid_result.shape
        x_min, x_max = xi.min(), xi.max()
        y_min, y_max = yi.min(), yi.max()
        pixel_size = grid_res # Tamanho do pixel em metros


        #Criar camada única com data e hora
        agora = datetime.now()
        timestamp = agora.strftime("%Y%m%d_%H%M%S")  # 20250505_153022
        nome_raster_saida = f"raster_ppv_{timestamp}.tif"
        # Caminho temporário onde o raster será salvo
        raster_path = os.path.join(tempfile.gettempdir(), nome_raster_saida)

        # Criação do GeoTIFF com GDAL
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(raster_path, ncols, nrows, 1, gdal.GDT_Float32)
        dataset.GetRasterBand(1).WriteArray(grid_result[::-1])  # inverte eixo Y (imagem)#-0-

        #-0- Escreve os dados no raster — sem inversão vertical, para manter orientação correta dos valores
        #dataset.GetRasterBand(1).WriteArray(grid_result)

        # Define a transformação geoespacial: origem no canto superior esquerdo e resolução
        # O valor negativo no eixo Y indica que os pixels crescem de cima para baixo, como esperado pelo QGIS
        dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

        # Define o sistema de coordenadas do raster com base na camada base
        srs = osr.SpatialReference()
        srs.ImportFromWkt(camada_base.crs().toWkt())
        dataset.SetProjection(srs.ExportToWkt())

        # Finaliza e salva o raster
        dataset.FlushCache()
        dataset = None

        # Conclui barra de progresso
        uii.progressBar_forPonto.setValue(100)

        log(f"✅ Raster de PPV salvo em: {raster_path}")
        
        # Carrega raster no QGIS
        raster_layer = QgsRasterLayer(raster_path, "Raster PPV")
        QgsProject.instance().addMapLayer(raster_layer)

        # === Geração de curvas de nível (contorno) ============================================
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
            log(f"Campo de espaçamento entre linhas vazio. Usando valor padrão ({intervalo} mm/s).")
            intervalo = float(0.5)
        
        #Criar camada única com data e hora
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
            'ITERATIONS': 3,  # número de iterações (2–5 costuma ser suficiente)
            'OFFSET': 0.25,   # fator de suavização (ajuste conforme necessário)
            'MAX_ANGLE': 180, # ângulo máximo entre segmentos (opcional)
            'OUTPUT': output_curvasS
        }, feedback=QgsProcessingFeedback())

        log(f"✅ Curvas de isovalores salvas em: {output_curvasS}")
        
        
        # Adiciona as curvas ao projeto
        curvas_layer = QgsVectorLayer(output_curvasS, "Curvas PPV", "ogr")
        QgsProject.instance().addMapLayer(curvas_layer)

        # ⚠️ Força carregamento dos campos para garantir acesso ao campo 'PPV'
        curvas_layer.dataProvider().reloadData()
        #-----------------------------------------------------------------------
        
        # Arredonda e aplica os valores do campo 'PPV'
        arredonda_campo(curvas_layer, 'PPV')
    except Exception as e: 
        tb = traceback.format_exc()                        
        log(f"⚠️ Erro na curva de isovalores: {e} \nTraceback:\n{tb}")

# Cria o indice espacial da camada de topografia
def indice_espacial_topo(ui, camada_topo, camada_topo3D, campo_z, log):
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
        for feat in camada_topo.getFeatures():
            # ProgressBar
            contadorBar += 1
            if feats_total > 0 and contadorBar % max((feats_total // 100), 1) == 0:
                progresso = int((contadorBar / feats_total) * 100)
                ui.progressBar_forPonto.setValue(min(progresso, 98))
                QCoreApplication.processEvents()
            
            geom = feat.geometry()
            if not geom:
                continue 
            for point in geom.vertices():
                repetir += 1
                try:
                    x, y = point.x(), point.y()
                    if camada_topo3D == True:
                        z = point.z()
                    else:
                        if campo_z:
                            z = float(feat[campo_z])
                        else:
                            z = obter_z_do_furo_mais_proximo(x, y, ui.resultados_processadosFsim, log)
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
        
        #Executa o processamento do plano de simulação
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
        layer_vertices, indice_topo, dicionario_vertices = indice_espacial_topo(ui, camada_topo, camada_topo3D, campo_z, log)
        #-----------------------------------------------------------------------------------

        # Gera a malha de simulação
        malha = gerar_malha(geometria_area)
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
            tolerancia = 10 # Tolerancia p/ caixa de busca.
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
                        
                tolerancia += 10    
                
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
            #xi_f, yi_f = float(xi), float(yi)
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

                if menor_dist_raiz == 0:
                    menor_dist_raiz = 0.1  # evita divisão por zero
                
                ppv = a * (carga ** b) * (menor_dist_raiz ** c)
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
        
        
        # Progress Bar até esse ponto
        ui.progressBar_forPonto.setVisible(True)
        ui.progressBar_forPonto.setValue(0)
        gerar_curvas_isovalores(
            pontos_ppv, valores_ppv,
            camada_topo,
            "raster_ppv.tif",
            "curvas_ppv",
            log,
            ui
        )
        
        '''# Interpolação
        xs = sorted(set([p[0] for p in pontos_ppv]))
        ys = sorted(set([p[1] for p in pontos_ppv]))
        zs = sorted(set([p[2] for p in pontos_ppv]))
        grid_z = interpolar_ppv(pontos_ppv, valores_ppv, xs, ys, zs, ui)
        
        #Debug
        log(f"Dados grid: {grid_z.shape}, xs: {len(xs)}, ys: {len(ys)}, zs: {len(zs) if zs else 'N/A'}")
        log(f"Coordenadas: xs={xs[:5]} ys={ys[:5]} zs={zs[:5] if zs else 'N/A'}")
        
        # Criação do shapefile interpolado
        salvar_shapefile(grid_z, xs, ys, camada_zcriticas.crs().authid(), zs,  "simulacao_ppv.shp")

        if vib_max > 0:
            grid_filtrado = np.where(grid_z > vib_max, grid_z, np.nan)
            salvar_shapefile(grid_filtrado, xs, ys, camada_zcriticas.crs().authid(), "simulacao_vib_max.shp")

        if carga_substituta > 0:
            salvar_shapefile(grid_z, xs, ys, camada_zcriticas.crs().authid(), "simulacao_carga_substituta.shp")
        
        # Progress Bar até esse ponto
        #ui.progressBar.setValue(100)
        
        log("Simulação concluída com sucesso.")'''

    except Exception as e:     
        tb = traceback.format_exc()                              
        #if contErro[0] <= vezesLog:
        log(f"⚠️ Erro na simulação: {e} \nTraceback:\n{tb}")
        #log(f"Erro na simulação: {str(e)}")


