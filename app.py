from datos import modelFeatures #simula una DB
from flask import Flask, jsonify, request, make_response
import pickle
import sklearn
import numpy as np
import pandas as pd

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>> declaramos el servidor
app = Flask(__name__)

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>> MAIN
def main():
    app.run(debug=True)

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>> Empezamoos a crear las rutas y los metodos para la REST API
@app.route('/ping')  # us GET por defecto
def ping():
    return jsonify({"mensaje": "pong"})

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>> Obtener todos los objetos de la lista
@app.route('/clientFeatures')
def getProdutcs():
    # estructura del objeto
    return jsonify({"clientes": modelFeatures, "mensaje": "Aqui los features por clientes"})

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>  Obtener un objeto de la lista
# esta ruta recibe paramtros para extraer info del objeto de forma dinamica

def predict(**objeto):
    features = dict(objeto)
    mF_df=pd.DataFrame(columns=(list(features.keys())))
    mF_df.loc[len(mF_df)]=list(features.values())
    # print(">>>>> Dataframe",mF_df)
    # cargo modelo para probar
    model = pickle.load(open('VF_model_rf.pkl', 'rb'))
    pred=model.predict(mF_df)
    print("Resultado",pred[0])
    return int(pred[0])

# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>> Predecir y agregar
@app.route('/predict', methods=['POST'])
def addObj():
    newObjDict = request.json
    # print("recibi: ",newObjDict)
    # print("---------",type(newObjDict))
    # ! esto no modifica en memoria debe ser sustituido por una DB
    # modelFeatures.append(newObjDict)
    p=predict(**newObjDict)
    return jsonify({'pred': p})

# Mensaje de bienvenida
@app.route('/')
def index():
    return "<h1>Servidor AI 4 Secure on!</h1>"

# ejecucion del servidor
if __name__ == '__main__':
    main()
