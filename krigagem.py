#Versão2 - Interpola apena uma variável ===============================================================================
import numpy as np
from pykrige.ok import OrdinaryKriging
from PyQt5.QtCore import QCoreApplication
from scipy.spatial import cKDTree

def krigagemF(nugget, sill, range_, modelo, exponent, scale, x, y, z, v, grid_x, grid_y, uii, tolerancia=100, log=print):
    """
    Interpola os valores v usando Krigagem Ordinária 2D com busca local.

    Parâmetros:
        nugget, sill, range_, modelo: parâmetros do variograma
        exponent e scale: usado apenas se modelo for 'power'
        x, y, z: coordenadas dos pontos amostrados
        v: valores amostrados (PPV, por exemplo)
        grid_x, grid_y: malha gerada com np.meshgrid
        uii: interface para atualizar a barra de progresso
        tolerancia: raio de busca em metros
        log: função de log (print padrão)

    Retorno:
        grid_result: matriz 2D com os valores interpolados
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

    # Inicializa matriz de resultados com NaN
    grid_result = np.full(grid_x.shape, np.nan)
    #feats_total = len(xi) * len(yi)
    feats_total = grid_x.shape[0] * grid_x.shape[1]
    contadorBar = 0


    coords = np.column_stack((x, y, z))
    tree = cKDTree(coords)
    
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            x0 = grid_x[i, j]
            y0 = grid_y[i, j]
            z0 = cota(tolerancia, x, y, z, x0, y0)
            if z0 == None or np.isnan(z0):
                continue
            
            # Filtra pontos próximos dentro do raio de tolerância--------------------------
            idxs = tree.query_ball_point([x0, y0, z0], r=50) # vizinhos dentro do raio r
            
            if len(idxs) < 3:
                dists, idxs_nn = tree.query([x0, y0, z0], k=3) #pega os 3 mais próximo, independente da distancia
                if np.isscalar(idxs_nn):
                    idxs_nn = [idxs_nn]
                idxs = list(idxs_nn)
            #------------------------------------------------------------------------------

            # converte índices em arrays locais
            x_local = x[idxs]
            y_local = y[idxs]
            v_local = v[idxs]
            
            variogram_params = {}

            # Define os parâmetros do variograma com base no modelo
            if modelo == "power":
                    scale_local = scale  # não sobrescreve scale externo
                    if scale_local is None:
                        # fallback usando distância máxima local (apenas em XY)
                        dists_xy_local = np.sqrt((x_local - x0)**2 + (y_local - y0)**2)
                        h_max = np.max(dists_xy_local) if dists_xy_local.size > 0 else 0.0
                        if h_max > 0:
                            scale_local = (sill - nugget) / (h_max ** exponent)
                        else:
                            scale_local = max((sill - nugget), 1e-12)
                    variogram_params = {"nugget": nugget, "scale": scale_local, "exponent": exponent}

            elif modelo == "linear":
                # Define 'slope' com base em (sill - nugget) / range_
                slope = (sill - nugget) / range_ if range_ != 0 else 1e-6
                variogram_params = {"nugget": nugget, "slope": slope}
            
            else:
                variogram_params = {"nugget": nugget, "sill": sill, "range": range_}
                

            try:
                OK_local = OrdinaryKriging(
                    x_local, y_local, v_local,
                    variogram_model=modelo,
                    variogram_parameters=variogram_params,
                    verbose=False,
                    enable_plotting=False
                )

                valor_interp, _ = OK_local.execute('points', [x0], [y0])
                grid_result[i, j] = valor_interp[0]

            except Exception as e:
                log(f"⚠️ Falha ao interpolar ({x0:.1f}, {y0:.1f}): {e}")
                grid_result[i, j] = np.nan

            contadorBar += 1
            # Atualiza barra de progresso
            if feats_total > 0 and contadorBar % max((feats_total // 100), 1) == 0:
                progresso = int((contadorBar / feats_total) * 100)
                uii.progressBar_forPonto.setValue(min(progresso, 100))
                QCoreApplication.processEvents()

    return grid_result

'''



#Versão3 - Interpola mais de uma variável e acresce ao grid =====================================================

import numpy as np
from pykrige.ok import OrdinaryKriging
from PyQt5.QtCore import QCoreApplication

def krigagemF(nugget, sill, range_, modelo, exponent, scale,
              x, y, z, v, grid_x, grid_y, uii,
              tolerancia=100, include_z_as_var=False, log=print):
    """
    Interpola usando Krigagem Ordinária 2D com busca local.
    - x,y,z: coordenadas dos pontos (z aqui é coordenada vertical)
    - v: pode ser 1D (n_points,) ou 2D (n_points, n_variaveis)
    - include_z_as_var: se True, inclui a coordenada z como uma variável adicional a ser interpolada
    - Retorna: se 1 variável -> array 2D (nx,ny); se n_variaveis -> array 3D (nx, ny, n_vars)
    """
    # validações e preparação das variáveis a serem interpoladas
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    v = np.asarray(v)

    if v.ndim == 1:
        vars_all = v.reshape(-1, 1)  # (n_points, 1)
        var_names = ["v"]
    elif v.ndim == 2:
        vars_all = v.copy()          # (n_points, n_vars)
        var_names = [f"var{i}" for i in range(vars_all.shape[1])]
    else:
        raise ValueError("v deve ser 1D ou 2D (n_points x n_variaveis)")

    if include_z_as_var:
        # acrescenta a coordenada z como última coluna nas variáveis a interpolar
        vars_all = np.column_stack([vars_all, z])
        var_names.append("z_coord")

    n_vars = vars_all.shape[1]

    # Inicializa resultado: 2D se n_vars==1, senão 3D (nx,ny,n_vars)
    nx, ny = grid_x.shape
    if n_vars == 1:
        grid_result = np.full((nx, ny), np.nan)
    else:
        grid_result = np.full((nx, ny, n_vars), np.nan)

    feats_total = nx * ny
    contadorBar = 0

    # função interna de cálculo de cota
    def cota_local(tolerancia_local, x_arr, y_arr, z_arr, x0, y0):
        raioBusca = tolerancia_local
        dists_2d = np.sqrt((x_arr - x0)**2 + (y_arr - y0)**2)
        minVizinho = 0

        while minVizinho <= 5:
            mask_local = dists_2d <= raioBusca
            minVizinho = np.sum(mask_local)
            raioBusca += 20
            if raioBusca >= 500:
                break

        if raioBusca > 500 and minVizinho <= 5:
            return None

        dists_vizinhos = dists_2d[mask_local]
        z_vizinhos = z_arr[mask_local]

        if dists_vizinhos.size == 0:
            return None

        if np.any(dists_vizinhos == 0):
            return np.mean(z_vizinhos[dists_vizinhos == 0])
        else:
            pesos = 1.0 / (dists_vizinhos**2 + 1e-12)
            return np.sum(pesos * z_vizinhos) / np.sum(pesos)

    # loop sobre a grade
    for i in range(nx):
        for j in range(ny):
            x0 = grid_x[i, j]
            y0 = grid_y[i, j]
            z0 = cota_local(tolerancia, x, y, z, x0, y0)
            if z0 is None or np.isnan(z0):
                continue

            # seleção de vizinhos usando distância 3D (x,y,z coordenada)
            dists3 = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
            mask = dists3 <= tolerancia

            # garante pelo menos 3 pontos
            if np.sum(mask) < 3:
                idxs = np.argsort(dists3)[:3]
                mask = np.zeros_like(dists3, dtype=bool)
                mask[idxs] = True

            x_local = x[mask]
            y_local = y[mask]
            vars_local = vars_all[mask, :]  # (n_neighbors, n_vars)

            # para cada variável, roda krigagem localmente
            for k in range(n_vars):
                v_local = vars_local[:, k]

                # monta variogram_params
                if modelo == "power":
                    scale_local = scale  # não sobrescreve scale externo
                    if scale_local is None:
                        # fallback usando distância máxima local (apenas em XY)
                        dists_xy_local = np.sqrt((x_local - x0)**2 + (y_local - y0)**2)
                        h_max = np.max(dists_xy_local) if dists_xy_local.size > 0 else 0.0
                        if h_max > 0:
                            scale_local = (sill - nugget) / (h_max ** exponent)
                        else:
                            scale_local = max((sill - nugget), 1e-12)
                    variogram_params = {"nugget": nugget, "scale": scale_local, "exponent": exponent}

                elif modelo == "linear":
                    slope = (sill - nugget) / range_ if range_ != 0 else 1e-6
                    variogram_params = {"nugget": nugget, "slope": slope}

                else:
                    variogram_params = {"nugget": nugget, "sill": sill, "range": range_}

                try:
                    # cria OK local e executa em um ponto
                    OK_local = OrdinaryKriging(
                        x_local, y_local, v_local,
                        variogram_model=modelo,
                        variogram_parameters=variogram_params,
                        verbose=False,
                        enable_plotting=False
                    )

                    valor_interp, _ = OK_local.execute('points', [x0], [y0])
                    out_val = valor_interp[0]

                except Exception as e:
                    log(f"⚠️ Falha ao interpolar {var_names[k]} em ({x0:.3f}, {y0:.3f}): {e}")
                    out_val = np.nan

                # grava resultado no grid_result (2D ou 3D conforme n_vars)
                if n_vars == 1:
                    grid_result[i, j] = out_val
                else:
                    grid_result[i, j, k] = out_val

            contadorBar += 1
            if feats_total > 0 and contadorBar % max((feats_total // 100), 1) == 0:
                progresso = int((contadorBar / feats_total) * 100)
                try:
                    uii.progressBar_forPonto.setValue(min(progresso, 100))
                    QCoreApplication.processEvents()
                except Exception:
                    # se uii não tiver a barra, ignora
                    pass

    return grid_result
'''