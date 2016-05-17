from bottle import route, run, request, response

import sys
sys.path.append('lib')
import predictor

@route('/predict')
def predict():
    source    = request.query.source
    sentences = predictor.predict(source)
    response.set_header('Access-Control-Allow-Origin', '*')
    return {"sentences": sentences}

run(host='localhost', port=3111, debug=True)
