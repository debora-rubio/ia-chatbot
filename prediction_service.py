#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serviço Flask para fazer predições usando o modelo treinado
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Permitir requisições do front-end

# Caminhos dos arquivos
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')

# Variáveis globais para o modelo
modelo = None
encoders = None
y_encoder = None
feature_names = None

def aplicar_replace(dados):
    """Converte valores categóricos para números usando mapeamento específico."""
    dados_encoded = dados.copy()
    
    # Mapeamento de valores categóricos para números
    mapeamentos = {
        # Sim/Não
        'Sim': 1,
        'Não': 2,
        
        # Finalidade
        'Industrial': 1,
        'Comercial': 2,
        'Doméstico': 3,
        
        # Capacidade (Kg por minuto)
        'Até 1Kg': 1,
        'Até 6.5Kg': 2,
        'Até 9Kg': 3,
        'Acima de 10Kg': 4,
        
        # Voltagem
        '127V': 1,
        '220V': 2,
        'Trifásico': 3,
        
        # Tipo de material
        'Embutidos': 1,
        'Carne, Frango': 2,
        'Diversos (Castanhas, Frutas, Graõs, Etc)': 3,
        
        # Orçamento
        'Até R$ 2,500,00': 1,
        'Até R$ 15,000,00': 2,
        'Acima de R$ 15,000,00': 3,
        
        # Potência
        'Até 0,25kW': 1,
        'Até 2,2kW': 2,
        'Até 5,5kW': 3,
        'Até 7,5kW': 4,
    }
    
    # Aplica o replace em todas as colunas
    for col in dados_encoded.columns:
        dados_encoded[col] = dados_encoded[col].replace(mapeamentos)
    
    return dados_encoded

def carregar_modelo():
    """Carrega o modelo e os encoders"""
    global modelo, encoders, y_encoder, feature_names
    
    if modelo is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute train_model.py primeiro.")
        
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(f"Encoders não encontrados em {ENCODERS_PATH}. Execute train_model.py primeiro.")
        
        print("📦 Carregando modelo...")
        with open(MODEL_PATH, 'rb') as f:
            modelo = pickle.load(f)
        
        print("📦 Carregando encoders...")
        with open(ENCODERS_PATH, 'rb') as f:
            encoders_data = pickle.load(f)
            encoders = encoders_data['label_encoders']
            y_encoder = encoders_data['y_encoder']
            feature_names = encoders_data['feature_names']
        
        print("✅ Modelo carregado com sucesso!")
    
    return modelo, encoders, y_encoder, feature_names

def extrair_regras_decisao(modelo, dados_encoded, feature_names, respostas_originais):
    """
    Extrai as regras da árvore de decisão que levaram à predição
    
    Args:
        modelo: Modelo de árvore de decisão treinado
        dados_encoded: DataFrame com os dados codificados
        feature_names: Lista com os nomes das features
        respostas_originais: Dict com as respostas originais (não codificadas)
    
    Returns:
        list: Lista de regras (dicts com 'pergunta', 'resposta', 'importancia')
    """
    regras = []
    
    try:
        # Obter o caminho da decisão na árvore
        decision_path = modelo.decision_path(dados_encoded)
        
        # Obter o nó folha (último nó do caminho)
        leaf_id = modelo.apply(dados_encoded)[0]
        
        # Obter o caminho até a folha
        node_indicator = decision_path.toarray()[0]
        path_nodes = [i for i, val in enumerate(node_indicator) if val == 1]
        
        # Extrair as features mais importantes do caminho
        # Vamos pegar os nós internos (não folhas) que foram visitados
        feature_importances = {}
        
        # Calcular a profundidade de cada nó
        def get_node_depth(node_id, depth=0):
            """Calcula a profundidade de um nó na árvore"""
            if node_id == 0:
                return depth
            # Encontrar o pai do nó
            for i in range(len(modelo.tree_.children_left)):
                if modelo.tree_.children_left[i] == node_id or modelo.tree_.children_right[i] == node_id:
                    return get_node_depth(i, depth + 1)
            return depth
        
        for node_id in path_nodes:
            if node_id == leaf_id:
                continue  # Pular o nó folha
            
            # Obter a feature usada neste nó
            feature_idx = modelo.tree_.feature[node_id]
            if feature_idx >= 0:  # -2 indica nó folha
                feature_name = feature_names[feature_idx]
                
                # Calcular importância baseada na profundidade (nós mais próximos da raiz são mais importantes)
                node_depth = get_node_depth(node_id)
                n_samples = modelo.tree_.n_node_samples[node_id]
                
                # Calcular score de importância (menor profundidade = maior importância)
                importance_score = (100 - node_depth * 10) + (n_samples / 10)
                
                if feature_name not in feature_importances or importance_score > feature_importances[feature_name]['score']:
                    feature_importances[feature_name] = {
                        'score': importance_score,
                        'depth': node_depth,
                        'n_samples': n_samples
                    }
        
        # Ordenar por score de importância (maior score = mais importante)
        sorted_features = sorted(feature_importances.items(), 
                                key=lambda x: x[1]['score'], 
                                reverse=True)
        
        # Converter para formato legível
        for idx, (feature_name, info) in enumerate(sorted_features[:7]):  # Top 7 mais importantes
            # Encontrar a resposta original correspondente
            resposta_original = respostas_originais.get(feature_name, '')
            
            if resposta_original:
                # Determinar importância: primeiras 3 são alta, resto média
                importancia = 'alta' if idx < 3 else 'média'
                regras.append({
                    'pergunta': feature_name,
                    'resposta': resposta_original,
                    'importancia': importancia
                })
    
    except Exception as e:
        print(f"⚠️  Erro ao extrair regras: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: retornar todas as respostas como regras, ordenadas
        todas_respostas = list(respostas_originais.items())
        for idx, (pergunta, resposta) in enumerate(todas_respostas[:7]):
            regras.append({
                'pergunta': pergunta,
                'resposta': resposta,
                'importancia': 'alta' if idx < 3 else 'média'
            })
    
    return regras

def prever_produto(respostas):
    """
    Faz a predição do produto baseado nas respostas e extrai as regras de decisão
    
    Args:
        respostas: dict com as respostas onde as chaves são os textos das perguntas:
        {
            'Para qual finalidade pretende usar o moedor?': 'Doméstico',
            'Quantos quilos precisa moer por minuto?': 'Até 1Kg',
            ...
        }
    
    Returns:
        dict: {
            'produto': str,  # Nome do produto recomendado
            'regras': list    # Lista de regras que levaram à decisão
        }
    """
    modelo, encoders, y_encoder, feature_names = carregar_modelo()
    
    # Garantir que as colunas estejam na ordem correta
    respostas_ordenadas = {col: respostas.get(col, '') for col in feature_names}
    
    # Criar DataFrame com as respostas na ordem correta
    dados = pd.DataFrame([respostas_ordenadas], columns=feature_names)
    
    # Aplicar replace
    dados_encoded = aplicar_replace(dados)
    
    # Aplicar LabelEncoder nas colunas que ainda são strings
    for col in dados_encoded.columns:
        if dados_encoded[col].dtype == 'object' and col in encoders:
            try:
                dados_encoded[col] = encoders[col].transform(dados_encoded[col])
            except ValueError:
                # Se o valor não estiver no encoder, usar o valor mais próximo
                dados_encoded[col] = 0
    
    # Fazer predição
    predicao_encoded = modelo.predict(dados_encoded)
    produto = y_encoder.inverse_transform(predicao_encoded)[0]
    
    # Extrair regras de decisão
    regras = extrair_regras_decisao(modelo, dados_encoded, feature_names, respostas)
    
    return {
        'produto': produto,
        'regras': regras
    }

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    try:
        carregar_modelo()
        return jsonify({
            'status': 'ok',
            'message': 'Serviço de predição está funcionando'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para fazer predição de produto"""
    try:
        data = request.json
        
        # Validar dados recebidos
        if not data:
            return jsonify({
                'success': False,
                'message': 'Dados não fornecidos'
            }), 400
        
        # Mapear respostas do chat para o formato esperado
        # O formato esperado é um dict com as chaves correspondentes às perguntas
        respostas_formatadas = {
            'Para qual finalidade pretende usar o moedor?': data.get('finalidade', ''),
            'Quantos quilos precisa moer por minuto?': data.get('quantidade', ''),
            'Qual é a voltagem que pretende utilizar?': data.get('voltagem', ''),
            'O que irá moer?': data.get('tipo_material', ''),
            'Prefere modelo mais fácil de limpar?': data.get('facil_limpeza', ''),
            'Ruído é um fator importante?': data.get('ruido_importante', ''),
            'O espaço físico é limitado?': data.get('espaco_limitado', ''),
            'Qual é a faixa de orçamento?': data.get('orcamento', ''),
            'Deseja função de remoagem?': data.get('remoagem', ''),
            'Potência desejada': data.get('potencia', '')
        }
        
        # Verificar se todas as respostas foram fornecidas
        valores_vazios = [k for k, v in respostas_formatadas.items() if not v]
        if valores_vazios:
            return jsonify({
                'success': False,
                'message': f'Respostas faltando: {", ".join(valores_vazios)}'
            }), 400
        
        # Fazer predição
        resultado = prever_produto(respostas_formatadas)
        
        return jsonify({
            'success': True,
            'produto': resultado['produto'],
            'regras': resultado['regras'],
            'respostas': respostas_formatadas
        })
        
    except Exception as e:
        print(f"❌ Erro ao fazer predição: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro ao fazer predição: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Iniciando serviço de predição...")
    print(f"📁 Diretório do modelo: {MODEL_DIR}")
    
    # Tentar carregar o modelo na inicialização
    try:
        carregar_modelo()
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível carregar o modelo na inicialização: {e}")
        print("⚠️  O modelo será carregado na primeira requisição")
    
    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=True)

