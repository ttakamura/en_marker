from bottle import route, run, request, response

@route('/predict')
def predict():
    source = request.query.source
    response.set_header('Access-Control-Allow-Origin', '*')
    return {"source": source}

run(host='localhost', port=3111, debug=True)
