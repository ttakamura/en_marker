import runner
import mark
from nltk.tokenize import sent_tokenize

model_prefix = 'model/current'
encdec, opt, conf = runner.load(model_prefix)

def split(source_text):
    return sent_tokenize(source_text)

def predict(source_text):
    sources   = split(source_text)
    sentences = []
    for source in sources:
        batch, hyp  = runner.predict(conf, encdec, source)
        x           = conf.corpus.tokenize(source, cleanup_tag=False)
        t, y        = hyp[0]
        annotations = mark.decoded_vec_to_hash(y)
        result      = []
        for i in range(len(x)):
            result.append({
                "source": x[i],
                "annotation": annotations[i]
            })
        sentences.append(result)
    return sentences
