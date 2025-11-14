from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui'  # Isso é necessário para session funcionar
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Criar pasta de uploads se não existir
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Adicionar filtros personalizados ao Jinja2
@app.template_filter('number_format')
def number_format(value):
    """Formata números com separadores de milhar"""
    try:
        return f"{int(value):,}".replace(",", ".")
    except (ValueError, TypeError):
        return value

@app.template_filter('currency_format')
def currency_format(value):
    """Formata valores monetários"""
    try:
        return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return value

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'csvFile' not in request.files:
            flash('Nenhum arquivo enviado', 'error')
            return redirect(url_for('index'))
        
        file = request.files['csvFile']
        
        if file.filename == '':
            flash('Nenhum arquivo selecionado', 'error')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            # Salvar arquivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Processar CSV
            df = pd.read_csv(filepath)
            
            # Validar campos obrigatórios
            required_columns = [
                'marca', 'modelo', 'ano_fabricacao', 'ano_modelo', 
                'valor_carro', 'quantidade_vendas', 
                'quantidade_procuras_nao_vendidas', 'total_procuras', 
                'valor_total_vendas', 'motorizacao', 'potencia_cv', 
                'tipo_cambio', 'combustivel', 'categoria', 'consumo_km_l'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                flash(f'Campos faltando: {missing_columns}', 'error')
                return redirect(url_for('index'))
            
            # Verificar se há valores nulos nos campos obrigatórios
            null_check = df[required_columns].isnull().any()
            columns_with_nulls = null_check[null_check == True].index.tolist()
            if columns_with_nulls:
                flash(f'Campos com valores nulos: {columns_with_nulls}', 'error')
                return redirect(url_for('index'))
            
            # Converter colunas para os tipos corretos
            df = convert_column_types(df)
            
            # Realizar análises
            analysis_results = perform_data_analysis(df)
            forecast_results = generate_forecasts(df)
            
            # Gerar gráficos
            charts = generate_charts(df, analysis_results, forecast_results)
            
            # Converter dados para lista
            data = df.to_dict('records')
            
            return render_template('results.html', 
                                 data=data,
                                 analysis=analysis_results,
                                 forecast=forecast_results,
                                 charts=charts,
                                 total_records=len(data))
        
        else:
            flash('Arquivo deve ser CSV', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Erro ao processar arquivo: {str(e)}', 'error')
        return redirect(url_for('index'))
    
@app.route('/ml_training', methods=['GET', 'POST'])
def ml_training():
    """Página de treinamento de machine learning"""
    
    if request.method == 'POST':
        try:
            # Verificar qual ação foi solicitada
            action = request.form.get('action')
            
            # Se for previsão e já temos modelo treinado, pular o treinamento
            if action == 'prever' and session.get('ml_model_trained'):
                # Usar dados da sessão para pular o treinamento
                return render_template('ml_training.html', 
                                     resultados_treinamento=True,
                                     nome_modelo=session.get('nome_modelo'),
                                     rmse=session.get('rmse'),
                                     r2=session.get('r2'),
                                     chart_base64=session.get('chart_base64'),
                                     params=session.get('params'),
                                     previsao_realizada=True,
                                     valor_estimado=0,  # Placeholder
                                     quantidade_vendas=0,  # Placeholder
                                     valor_total_vendas=0,  # Placeholder
                                     dados_previsao={})  # Placeholder
            
            # Verificar se foi enviado arquivo para treinamento
            if 'mlFile' not in request.files:
                flash('Nenhum arquivo enviado para treinamento', 'error')
                return render_template('ml_training.html')
            
            file = request.files['mlFile']
            
            if file.filename == '':
                flash('Nenhum arquivo selecionado para treinamento', 'error')
                return render_template('ml_training.html')
            
            if file and allowed_file(file.filename):
                # Salvar arquivo de treinamento
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Carregar dados
                df = pd.read_csv(filepath)
                df = convert_column_types(df)
                
                # Obter parâmetros do formulário
                modelo_type = int(request.form.get('modelo_type', 1))
                
                # Selecionar colunas para o modelo
                categorical_cols = ['modelo', 'tipo_cambio', 'combustivel', 'categoria', 'motorizacao']
                numeric_cols = ['ano_fabricacao', 'ano_modelo', 'potencia_cv', 'consumo_km_l']
                
                # Verificar se todas as colunas necessárias existem
                required_ml_columns = categorical_cols + numeric_cols + ['valor_carro']
                missing_columns = [col for col in required_ml_columns if col not in df.columns]
                if missing_columns:
                    flash(f'Colunas faltando no arquivo para ML: {missing_columns}', 'error')
                    return render_template('ml_training.html')
                
                # Codificar variáveis categóricas
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                
                # Separar features (X) e alvo (y)
                features_columns = categorical_cols + numeric_cols
                X = df[features_columns]
                y = df['valor_carro']
                
                # Verificar se há dados suficientes
                if len(df) < 10:
                    flash('Arquivo muito pequeno para treinamento. Mínimo 10 registros.', 'error')
                    return render_template('ml_training.html')
                
                # Normalizar os dados
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Dividir em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Configurar modelo baseado na escolha
                if modelo_type == 1:
                    model = LinearRegression()
                    nome_modelo = "Regressão Linear"
                    params = {}
                    
                elif modelo_type == 2:
                    max_depth = int(request.form.get('max_depth', 5))
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                    nome_modelo = f"Árvore de Decisão (max_depth={max_depth})"
                    params = {'max_depth': max_depth}
                    
                elif modelo_type == 3:
                    n_estimators = int(request.form.get('n_estimators', 100))
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    nome_modelo = f"Random Forest (n_estimators={n_estimators})"
                    params = {'n_estimators': n_estimators}
                    
                elif modelo_type == 4:
                    n_neighbors = int(request.form.get('n_neighbors', 5))
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                    nome_modelo = f"KNN (n_neighbors={n_neighbors})"
                    params = {'n_neighbors': n_neighbors}
                    
                else:
                    flash('Modelo inválido selecionado', 'error')
                    return render_template('ml_training.html')
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer previsões
                y_pred = model.predict(X_test)
                
                # Avaliar modelo
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Gerar gráfico de desempenho
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel("Valor real")
                plt.ylabel("Valor previsto")
                plt.title(f"Desempenho do modelo - {nome_modelo}")
                
                # Converter gráfico para base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                plt.close()
                
                # Se foi solicitada uma previsão
                if action == 'prever':
                    # Coletar dados para previsão
                    dados_previsao = {
                        "modelo": request.form.get('modelo'),
                        "motorizacao": request.form.get('motorizacao'),
                        "tipo_cambio": request.form.get('tipo_cambio'),
                        "combustivel": request.form.get('combustivel'),
                        "categoria": request.form.get('categoria'),
                        "ano_fabricacao": int(request.form.get('ano_fabricacao')),
                        "ano_modelo": int(request.form.get('ano_modelo')),
                        "potencia_cv": float(request.form.get('potencia_cv')),
                        "consumo_km_l": float(request.form.get('consumo_km_l'))
                    }
                    
                    # Criar DataFrame para previsão
                    novo_df = pd.DataFrame([dados_previsao])
                    
                    # Aplicar transformações
                    for col in categorical_cols:
                        if col in label_encoders:
                            try:
                                novo_df[col] = label_encoders[col].transform([dados_previsao[col]])[0]
                            except ValueError:
                                # Se o valor não foi visto durante o treinamento, usar o primeiro valor
                                novo_df[col] = 0
                    
                    # Garantir que as colunas estão na ordem correta
                    novo_df = novo_df[features_columns]
                    
                    # Normalizar
                    novo_scaled = scaler.transform(novo_df)
                    
                    # Fazer previsão
                    predicao = model.predict(novo_scaled)[0]
                    
                    # Gerar quantidade de vendas estimada
                    import random
                    quantidade_vendas = random.randint(50, 1000)
                    valor_total_vendas = round(predicao * quantidade_vendas)
                    
                    # Salvar informações na sessão para futuras previsões
                    session['ml_model_trained'] = True
                    session['modelo_type'] = modelo_type
                    session['features_columns'] = features_columns
                    session['nome_modelo'] = nome_modelo
                    session['rmse'] = rmse
                    session['r2'] = r2
                    session['chart_base64'] = chart_base64
                    session['params'] = params
                    
                    return render_template('ml_training.html', 
                                         resultados_treinamento=True,
                                         nome_modelo=nome_modelo,
                                         rmse=rmse,
                                         r2=r2,
                                         chart_base64=chart_base64,
                                         params=params,
                                         previsao_realizada=True,
                                         valor_estimado=predicao,
                                         quantidade_vendas=quantidade_vendas,
                                         valor_total_vendas=valor_total_vendas,
                                         dados_previsao=dados_previsao)
                
                # Se foi apenas treinamento
                elif action == 'treinar':
                    # Salvar informações na sessão
                    session['ml_model_trained'] = True
                    session['modelo_type'] = modelo_type
                    session['features_columns'] = features_columns
                    session['nome_modelo'] = nome_modelo
                    session['rmse'] = rmse
                    session['r2'] = r2
                    session['chart_base64'] = chart_base64
                    session['params'] = params
                    
                    return render_template('ml_training.html', 
                                         resultados_treinamento=True,
                                         nome_modelo=nome_modelo,
                                         rmse=rmse,
                                         r2=r2,
                                         chart_base64=chart_base64,
                                         params=params)
            
            else:
                flash('Arquivo deve ser CSV', 'error')
                return render_template('ml_training.html')
                
        except Exception as e:
            flash(f'Erro no treinamento: {str(e)}', 'error')
            return render_template('ml_training.html')
    
    # GET request - mostrar página vazia
    return render_template('ml_training.html')

def convert_column_types(df):
    """Converte as colunas para os tipos corretos"""
    try:
        # Colunas numéricas
        numeric_columns = ['ano_fabricacao', 'ano_modelo', 'valor_carro', 'quantidade_vendas',
                          'quantidade_procuras_nao_vendidas', 'total_procuras', 'valor_total_vendas',
                          'potencia_cv', 'consumo_km_l']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Colunas de texto
        text_columns = ['marca', 'modelo', 'motorizacao', 'tipo_cambio', 'combustivel', 'categoria']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    except Exception as e:
        raise Exception(f"Erro na conversão de tipos: {str(e)}")

def generate_charts(df, analysis, forecast):
    """Gera gráficos como imagens base64"""
    charts = {}
    
    # Configurar estilo dos gráficos
    plt.style.use('default')
    
    # 1. Gráfico de Evolução de Vendas
    fig, ax = plt.subplots(figsize=(10, 6))
    anos = analysis['evolucao_temporal']['anos']
    vendas = analysis['evolucao_temporal']['vendas']
    
    ax.plot(anos, vendas, marker='o', linewidth=2, markersize=8, color='#36a2eb')
    ax.fill_between(anos, vendas, alpha=0.3, color='#36a2eb')
    ax.set_title('Evolução de Vendas por Ano', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano de Fabricação')
    ax.set_ylabel('Quantidade de Vendas')
    ax.grid(True, alpha=0.3)
    
    # Salvar como base64
    charts['sales_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Gráfico de Evolução de Faturamento
    fig, ax = plt.subplots(figsize=(10, 6))
    faturamento = analysis['evolucao_temporal']['faturamento']
    
    ax.bar(anos, faturamento, color='#4BC0C0', alpha=0.8)
    ax.set_title('Evolução de Faturamento por Ano', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano de Fabricação')
    ax.set_ylabel('Faturamento (R$)')
    ax.grid(True, alpha=0.3)
    
    # Formatar eixo Y em milhões
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1000000:.1f}M'))
    
    charts['revenue_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 3. Gráfico de Distribuição por Modelo (Top 8)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_models = analysis['top_10_modelos_vendas'][:8]
    modelos = [item['modelo'] for item in top_models]
    vendas_modelos = [item['vendas'] for item in top_models]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(modelos)))
    ax.pie(vendas_modelos, labels=modelos, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Distribuição de Vendas por Modelo (Top 8)', fontsize=14, fontweight='bold')
    
    charts['distribution_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 4. Gráfico de Preço Médio
    fig, ax = plt.subplots(figsize=(10, 6))
    preco_medio = analysis['evolucao_temporal']['preco_medio']
    
    ax.plot(anos, preco_medio, marker='s', linewidth=2, markersize=8, color='#FF6384')
    ax.fill_between(anos, preco_medio, alpha=0.3, color='#FF6384')
    ax.set_title('Evolução do Preço Médio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano de Fabricação')
    ax.set_ylabel('Preço Médio (R$)')
    ax.grid(True, alpha=0.3)
    
    # Formatar eixo Y em reais
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x:,.0f}'.replace(',', '.')))
    
    charts['price_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 5. Gráfico de Taxa de Conversão (Top 10)
    fig, ax = plt.subplots(figsize=(12, 6))
    modelos_taxa = []
    taxas = []
    
    for modelo, dados in analysis['analise_por_modelo'].items():
        if dados['taxa_conversao'] > 0:  # Ignorar taxas zero
            modelos_taxa.append(modelo)
            taxas.append(dados['taxa_conversao'])
    
    # Pegar top 10 por taxa de conversão
    indices = np.argsort(taxas)[-10:]
    modelos_taxa = [modelos_taxa[i] for i in indices]
    taxas = [taxas[i] for i in indices]
    
    bars = ax.barh(modelos_taxa, taxas, color='#9966FF', alpha=0.8)
    ax.set_title('Taxa de Conversão por Modelo (Top 10)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Taxa de Conversão (%)')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    charts['conversion_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 6. Gráfico de Previsão
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dados históricos
    anos_historicos = analysis['evolucao_temporal']['anos']
    vendas_historicas = analysis['evolucao_temporal']['vendas']
    
    # Dados de previsão
    anos_futuros = forecast['proximos_anos']
    vendas_previstas = forecast['previsao_geral']['vendas']
    
    # Plotar histórico
    ax.plot(anos_historicos, vendas_historicas, marker='o', linewidth=2, 
            markersize=8, color='#36a2eb', label='Histórico')
    ax.fill_between(anos_historicos, vendas_historicas, alpha=0.3, color='#36a2eb')
    
    # Plotar previsão
    ax.plot(anos_futuros, vendas_previstas, marker='s', linewidth=2, 
            markersize=8, color='#4CAF50', linestyle='--', label='Previsão')
    ax.fill_between(anos_futuros, vendas_previstas, alpha=0.3, color='#4CAF50')
    
    ax.set_title('Projeção de Vendas - Histórico vs Previsão', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade de Vendas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    charts['forecast_chart'] = fig_to_base64(fig)
    plt.close(fig)
    
    return charts

def fig_to_base64(fig):
    """Converte figura matplotlib para base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

def perform_data_analysis(df):
    """Realiza análise completa dos dados"""
    
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

def generate_forecasts(df):
    """Gera previsões para os próximos 5 anos"""
    
    max_ano = df['ano_fabricacao'].max()
    proximos_anos = list(range(max_ano + 1, max_ano + 6))
    
    forecast = {
        'previsao_geral': {},
        'previsao_por_modelo': {},
        'metricas_modelo': {},
        'proximos_anos': proximos_anos
    }
    
    # Previsão por modelo (top 5)
    modelos_populares = df.groupby('modelo')['quantidade_vendas'].sum().nlargest(5).index
    
    for modelo in modelos_populares:
        modelo_data = df[df['modelo'] == modelo].copy()
        
        if len(modelo_data) < 2:
            continue
            
        X = modelo_data[['ano_fabricacao']].values
        y_vendas = modelo_data['quantidade_vendas'].values
        
        model_vendas = LinearRegression()
        model_vendas.fit(X, y_vendas)
        
        anos_futuros = np.array(proximos_anos).reshape(-1, 1)
        previsoes_vendas = [max(0, int(v)) for v in model_vendas.predict(anos_futuros)]
        
        forecast['previsao_por_modelo'][modelo] = {
            'vendas': previsoes_vendas,
            'precisao': float(model_vendas.score(X, y_vendas))
        }
    
    # Previsão geral
    dados_gerais = df.groupby('ano_fabricacao').agg({
        'quantidade_vendas': 'sum',
        'valor_total_vendas': 'sum'
    }).reset_index()
    
    if len(dados_gerais) >= 2:
        X_geral = dados_gerais[['ano_fabricacao']].values
        y_vendas_geral = dados_gerais['quantidade_vendas'].values
        
        model_vendas_geral = LinearRegression()
        model_vendas_geral.fit(X_geral, y_vendas_geral)
        
        anos_futuros_geral = np.array(proximos_anos).reshape(-1, 1)
        previsoes_vendas_geral = [max(0, int(v)) for v in model_vendas_geral.predict(anos_futuros_geral)]
        
        forecast['previsao_geral'] = {
            'vendas': previsoes_vendas_geral,
            'precisao': float(model_vendas_geral.score(X_geral, y_vendas_geral))
        }
    else:
        forecast['previsao_geral'] = {
            'vendas': [0] * len(proximos_anos),
            'precisao': 0.0
        }
    
    forecast['metricas_modelo'] = {
        'algoritmo': 'Regressão Linear',
        'descricao': 'Modelo baseado em tendências históricas'
    }
    
    return forecast

if __name__ == '__main__':
    app.run(debug=True, port=5000)