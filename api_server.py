from bottle import route, run, request, response

import sys
sys.path.append('lib')
import predictor

@route('/predict', method='POST')
def predict():
    source    = request.params.source
    sentences = predictor.predict(source)
    response.set_header('Access-Control-Allow-Origin', '*')
    return {"sentences": sentences}

run(host='localhost', port=3111, debug=True, reloader=True)
