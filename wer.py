#pip3 install jiwer
from jiwer import wer
def text_from_subtitle(filename='subtitle.vtt'):
    f = open(filename, 'r')
    fulltext = ''
    for line in f:
        if line[0] == '-':
            fulltext += line[1:-1]
    return fulltext


hypothesis = text_from_subtitle()
reference = text_from_subtitle('reference.vtt')
error = wer(reference, hypothesis)
print(error)