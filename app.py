from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'csvFile' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['csvFile']
        
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if file and file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
            
            required_columns = [
                'marca', 'modelo', 'ano_fabricacao', 'ano_modelo', 
                'valor_carro', 'quantidade_vendas', 
                'quantidade_procuras_nao_vendidas', 'total_procuras', 
                'valor_total_vendas'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({'error': f'Campos faltando: {missing_columns}'}), 400
            
            # Realizar análise de dados
            analysis_results = perform_data_analysis(df)
            
            # Converter dados para lista de dicionários
            data = df.to_dict('records')
            
            return jsonify({
                'success': True,
                'data': data,
                'total_records': len(data),
                'analysis': analysis_results
            })
        
        else:
            return jsonify({'error': 'Arquivo deve ser CSV'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500

def perform_data_analysis(df):
    
    # Análise Básica
    analysis = {
        'resumo_geral': {
            'total_vendas': int(df['quantidade_vendas'].sum()),
            'faturamento_total': float(df['valor_total_vendas'].sum()),
            'media_preco_carro': float(df['valor_carro'].mean()),
            'total_procuras': int(df['total_procuras'].sum()),
            'taxa_conversao_geral': float((df['quantidade_vendas'].sum() / df['total_procuras'].sum()) * 100)
        },
        
        'analise_por_ano': {},
        'analise_por_modelo': {},
        'top_10_modelos_vendas': [],
        'top_10_modelos_faturamento': [],
        'evolucao_temporal': {},
        'correlacoes': {}
    }
    
    # Análise por Ano de Fabricação
    for ano in sorted(df['ano_fabricacao'].unique()):
        dados_ano = df[df['ano_fabricacao'] == ano]
        analysis['analise_por_ano'][str(ano)] = {
            'vendas': int(dados_ano['quantidade_vendas'].sum()),
            'faturamento': float(dados_ano['valor_total_vendas'].sum()),
            'media_preco': float(dados_ano['valor_carro'].mean()),
            'taxa_conversao': float((dados_ano['quantidade_vendas'].sum() / dados_ano['total_procuras'].sum()) * 100)
        }
    
    # Análise por Modelo
    for modelo in df['modelo'].unique():
        dados_modelo = df[df['modelo'] == modelo]
        analysis['analise_por_modelo'][modelo] = {
            'vendas_totais': int(dados_modelo['quantidade_vendas'].sum()),
            'faturamento_total': float(dados_modelo['valor_total_vendas'].sum()),
            'media_preco': float(dados_modelo['valor_carro'].mean()),
            'procuras_totais': int(dados_modelo['total_procuras'].sum()),
            'taxa_conversao': float((dados_modelo['quantidade_vendas'].sum() / dados_modelo['total_procuras'].sum()) * 100)
        }
    
    # Top 10 Modelos por Vendas
    top_vendas = df.groupby('modelo')['quantidade_vendas'].sum().nlargest(10)
    analysis['top_10_modelos_vendas'] = [
        {'modelo': modelo, 'vendas': int(vendas)} 
        for modelo, vendas in top_vendas.items()
    ]
    
    # Top 10 Modelos por Faturamento
    top_faturamento = df.groupby('modelo')['valor_total_vendas'].sum().nlargest(10)
    analysis['top_10_modelos_faturamento'] = [
        {'modelo': modelo, 'faturamento': float(faturamento)} 
        for modelo, faturamento in top_faturamento.items()
    ]
    
    # Evolução Temporal
    evolucao = df.groupby('ano_fabricacao').agg({
        'quantidade_vendas': 'sum',
        'valor_total_vendas': 'sum',
        'valor_carro': 'mean'
    }).sort_index()
    
    analysis['evolucao_temporal'] = {
        'anos': [int(ano) for ano in evolucao.index],
        'vendas': [int(v) for v in evolucao['quantidade_vendas'].values],
        'faturamento': [float(f) for f in evolucao['valor_total_vendas'].values],
        'preco_medio': [float(p) for p in evolucao['valor_carro'].values]
    }
    
    # Correlações
    correlacao_data = df[['valor_carro', 'quantidade_vendas', 'total_procuras']].corr()
    analysis['correlacoes'] = {
        'preco_vs_vendas': float(correlacao_data.loc['valor_carro', 'quantidade_vendas']),
        'preco_vs_procura': float(correlacao_data.loc['valor_carro', 'total_procuras']),
        'vendas_vs_procura': float(correlacao_data.loc['quantidade_vendas', 'total_procuras'])
    }
    
    return analysis

if __name__ == '__main__':
    app.run(debug=True, port=5000)