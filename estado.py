# estado.py

def resetar_estado_plugin(plugin):
    plugin.resultados_processamento = None
    plugin.dados_regressao = None
    plugin.resultados_simulacao = None
    plugin.ui = None
    plugin.camadas_carregadas = []
    plugin.resultado_regressao = None
    plugin.dados_simulados = []
    plugin.raster_layer = None
    plugin.camada_furos = None
    plugin.camada_geofones = None
    plugin.camada_topografia = None
    plugin.camada_area_critica = None
    plugin.camada_furos_simular = None
    plugin.resultado_tabela = []
    plugin.resultado_regressao_dict = {}
    plugin.grid_points_layer = None
    plugin.interpolador = None
    plugin.campos_topografia = []
    plugin.z_por_coluna = False
    plugin.coluna_z = None
    plugin.tipo_interpolador = None
    plugin.equacao_atenuacao = None
    plugin.parametros = None
    plugin.malha = []
    plugin.campos_furos = []
    plugin.campo_carga = None
    plugin.campo_retardo = None
    plugin.campo_id_furo = None
    plugin.campo_z_furo = None
    plugin.campo_id_geofone = None
    plugin.campo_ppv = None
    plugin.campo_z_geofone = None
    plugin.espacamento = None
    plugin.distancias = []
    plugin.resultados_por_geofone = {}
    plugin.resultados_simulacao_detalhados = []

    print("🔁 Estado interno do plugin foi reiniciado.")

