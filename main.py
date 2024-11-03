from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado
modelo = joblib.load('modelo_random_forest.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()  # Recebe os dados em JSON
    
    # Converter os dados em DataFrame
    try:
        dados_df = pd.DataFrame([dados])

        # Converta as colunas numéricas que o modelo espera
        colunas_numericas = ['ano_fabricacao', 'quilometragem']  # Substitua pelas colunas numéricas reais
        for coluna in colunas_numericas:
            if coluna in dados_df.columns:
                dados_df[coluna] = pd.to_numeric(dados_df[coluna], errors='coerce')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Definir as variáveis que representam problemas
    variaveis_de_problemas = ['filtro_oleo', 'oleo_motor', 'filtro_ar', 'filtro_combustivel', 'vela_ignicao', 'fluido_freio', 'pastilhas_freio', 'embreagem']  # Coloque aqui os nomes das colunas que indicam problemas

    # Verificar se todas as variáveis de problemas estão em "0"
    if all(dados_df[variavel].iloc[0] == 0 for variavel in variaveis_de_problemas):
        predicted_cost = 0
    else:
        # Fazer a previsão se houver algum problema
        predicted_cost = modelo.predict(dados_df)[0]

    return jsonify({'predicted_cost': predicted_cost})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
