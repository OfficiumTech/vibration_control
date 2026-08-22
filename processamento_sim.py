import os
import matplotlib.pyplot as plt
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsWkbTypes
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton
import traceback


# Função que retorna a camada do projeto com o nome informado
def obter_camada_por_nome(nome):
    for camada in QgsProject.instance().mapLayers().values():
        if camada.name() == nome:
            return camada
    return None

# Encontra o nome real de um campo da camada a partir de possíveis nomes
def encontrar_nome_campo(campos, possiveis_nomes):
    for nome in possiveis_nomes:
        if nome in campos:
            return nome
    return None


# Verifica se a camada é 3D ou tenta encontrar campo Z na tabela de atributos
def verificar_3d_ou_campo_z(camada, nome_camada, avisos):
    if QgsWkbTypes.hasZ(camada.wkbType()):
        return None  # Geometria já é 3D, não precisa de campo Z

    # Lista de nomes padrão de campos Z
    nomes_possiveis = ["z", "Z", "Cota", "Elevação"]
    campos = [campo.name() for campo in camada.fields()]

    for nome in nomes_possiveis:
        if nome in campos:
            avisos.append(f"A camada '{nome_camada}' é 2D. Usando campo '{nome}' como cota Z.")
            return nome

    # Nenhum campo padrão encontrado, solicitar ao usuário
    campo_escolhido = dialogo_escolher_campo_z(camada, nome_camada)
    if campo_escolhido:
        avisos.append(f"A camada '{nome_camada}' é 2D. Usando campo '{campo_escolhido}' como cota Z.")
    else:
        avisos.append(f"A camada '{nome_camada}' é 2D. Nenhum campo Z será usado.")
    return campo_escolhido


# Cria um popup para o usuário escolher um campo Z ou seguir sem cota
def dialogo_escolher_campo_z(camada, nome_camada):
    dialog = QDialog()
    dialog.setWindowTitle(f"Selecionar campo Z - {nome_camada}")
    layout = QVBoxLayout()

    label = QLabel("A geometria desta camada é 2D. Selecione o campo da tabela que representa a cota Z:")
    layout.addWidget(label)

    combo = QComboBox()
    campos = [campo.name() for campo in camada.fields()]
    combo.addItems(campos)
    combo.addItem("Sem cota z")
    layout.addWidget(combo)

    botao_ok = QPushButton("OK")
    botao_ok.clicked.connect(dialog.accept)
    layout.addWidget(botao_ok)

    dialog.setLayout(layout)

    if dialog.exec_() == QDialog.Accepted:
        escolhido = combo.currentText()
        if escolhido == "Sem cota z":
            return None
        return escolhido
    return None


#Logs
def log(mensagem):
    QMessageBox.information(None, "Procesamento - Blaster Vibration Control", mensagem)


# Função principal
def processar_furoSim(ui):
    try:
        #Obter os nomes das camadas selecionadas nos comboBoxes
        nome_furos_simular = ui.comboBox_shpFuros_simular.currentText() #-0-

        #Obter as camadas reais do projeto pelo nome
        camada_furos_simular = obter_camada_por_nome(nome_furos_simular)

        simular = False
        if camada_furos_simular:#-0-
            simular = True
        
        # Verificar se as camadas são 3D e armazenar mensagens de aviso caso não sejam
        avisos_3d = []
                
        if simular:#-0-
            z_field_furos_simular = verificar_3d_ou_campo_z(camada_furos_simular, nome_furos_simular, avisos_3d)
            campos_furos_simular = [campo.name() for campo in camada_furos_simular.fields()]
            
            campo_seq_simular = encontrar_nome_campo(campos_furos_simular, ["Seq. D.", "Ret. (ms)", "seq. d.", "ret. (ms)", "Sequencia", "Sequência"])
            campo_carga_simular = encontrar_nome_campo(campos_furos_simular, ["Carga (KG)", "Cargas (KG)", "carga (kg)", "cargas (kg)", "Carga(KG)", "Cargas(KG)", "carga(kg)", "cargas(kg)" ])
            campo_id_furo_simular = encontrar_nome_campo(campos_furos_simular, ["Id", "id", "ID", "Identificação", "Ident", "Ponto", "PONTO"])
            
            if not all([campo_seq_simular, campo_carga_simular, campo_id_furo_simular]):
                raise Exception("Campos obrigatórios ausentes na tabela de atributos dos furos de simulação. Leia a aba 'HELP'.")

        if avisos_3d:
            QMessageBox.warning(None, "Aviso - Dados 2D", "\n".join(avisos_3d))
            
        #-0-Furos de simulação-----------------------------------------------------------------------------------
        # Se a camada de simulação estiver presente, também processa
        if simular:
            # Agrupar furos_simular por sequência de detonação
            grupos_simular = {}
            for f in camada_furos_simular.getFeatures():
                seq = f[campo_seq_simular]
                carga = f[campo_carga_simular]
                if seq not in grupos_simular:
                    grupos_simular[seq] = []
                grupos_simular[seq].append((f, carga))

            # Calcular carga total por grupo
            cargas_por_seq_simular = {seq: sum([c for _, c in furos]) for seq, furos in grupos_simular.items()}

            # Identificar a(s) sequência(s) com maior carga
            maior_carga_total_simular = max(cargas_por_seq_simular.values())
            seqs_maior_carga_simular = [seq for seq, carga in cargas_por_seq_simular.items() if carga == maior_carga_total_simular]
        
        
            # Para cada sequência com maior carga, pega os furos com maior carga individual
            resultadosFsim = []
            for seq in seqs_maior_carga_simular:
                furos_seq = grupos_simular[seq]
                
                for f, carga in furos_seq:
                    geom_f = f.geometry()
                    if geom_f.isEmpty() or QgsWkbTypes.geometryType(geom_f.wkbType()) != QgsWkbTypes.PointGeometry:
                        continue
                    pt_f = geom_f.constGet()
                    xf, yf = pt_f.x(), pt_f.y()
                    if QgsWkbTypes.hasZ(geom_f.wkbType()):
                        zf = pt_f.z()
                    else:
                        zf = f[z_field_furos_simular] if z_field_furos_simular else 0
                    resultadosFsim.append({
                        "ID Furo sim": f[campo_id_furo_simular],
                        "Total Carga / Seq.": maior_carga_total_simular,
                        # Remover a coluna "Seq. D. sim" se não for mais necessária:
                        # "Seq. D. sim": seq,
                        "Carga (KG) sim": carga,
                        "x do furo sim": xf,
                        "y do furo sim": yf,
                        "z do furo sim": zf
                    })
            # Camada de simulação - calculada e armazenada-------------------------------------------
            
            ui.resultados_processadosFsim = resultadosFsim
        
    except Exception as e:
        tb = traceback.format_exc()
        log(f"⚠️ Erro no Processamento do Plano Simulado: {e} \nTraceback:\n{tb}")
