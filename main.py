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
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Definir as variáveis que representam problemas
    variaveis_de_problemas = ['filtro_óleo', 'óleo_motor', 'filtro_ar', 'filtro_combustível', 'vela_ignição', 'fluido_freio', 'pastilhas_freio', 'embreagem']  # Coloque aqui os nomes das colunas que indicam problemas

    # Verificar se todas as variáveis de problemas estão em "0"
    if all(dados_df[variavel].iloc[0] == 0 for variavel in variaveis_de_problemas):
        predicted_cost = 0
    else:
        # Fazer a previsão se houver algum problema
        predicted_cost = modelo.predict(dados_df)[0]

    return jsonify({'predicted_cost': predicted_cost})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
