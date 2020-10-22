import spacy
import math

test = '''Seit der Industriellen Revolution verstärkt der Mensch den natürlichen Treibhauseffekt durch den Ausstoß von Treibhausgasen, wie messtechnisch belegt werden konnte. Seit 1990 ist der Strahlungsantrieb das heißt die Erwärmungswirkung auf das Klima durch langlebige Treibhausgase um 43 Prozent gestiegen. In der Klimatologie ist es heute Konsens, dass die gestiegene Konzentration der vom Menschen in die Erdatmosphäre freigesetzten Treibhausgase mit hoher Wahrscheinlichkeit die wichtigste Ursache der globalen Erwärmung ist, da ohne sie die gemessenen Temperaturen nicht zu erklären sind. Treibhausgase lassen die von der Sonne kommende kurzwellige Strahlung weitgehend ungehindert auf die Erde durch, absorbieren aber einen Großteil der von der Erde ausgestrahlten Infrarotstrahlung. Dadurch erwärmen sie sich und emittieren selbst Strahlung im langwelligen Bereich (vgl. Kirchhoffsches Strahlungsgesetz). Der in Richtung der Erdoberfläche gerichtete Strahlungsanteil wird als atmosphärische Gegenstrahlung bezeichnet. Im isotropen Fall wird die absorbierte Energie je zur Hälfte in Richtung Erde und Weltall abgestrahlt. Hierdurch erwärmt sich die Erdoberfläche stärker, als wenn allein die kurzwellige Strahlung der Sonne sie erwärmen würde. Das IPCC schätzt den Grad des wissenschaftlichen Verständnisses über die Wirkung von Treibhausgasen als hoch ein. Das Treibhausgas Wasserdampf trägt mit 36 bis 66 Prozent, Kohlenstoffdioxid mit 9 bis 26 Prozent und Methan mit 4 bis 9 Prozent zum natürlichen Treibhauseffekt bei. Die große Bandbreite erklärt sich folgendermaßen: Einerseits gibt es sowohl örtlich wie auch zeitlich große Schwankungen in der Konzentration dieser Gase. Zum anderen überlappen sich deren Absorptionsspektren. Beispiel: Strahlung, die von Wasserdampf bereits absorbiert wurde, kann von CO2 nicht mehr absorbiert werden. Das bedeutet, dass in einer Umgebung wie eisbedeckte Flächen oder Trockenwüste, in der Wasserdampf nur wenig zum Treibhauseffekt beiträgt, die übrigen Treibhausgase mehr zum Gesamttreibhauseffekt beitragen als in den feuchten Tropen. Da die genannten Treibhausgase natürliche Bestandteile der Atmosphäre sind, wird die von ihnen verursachte Temperaturerhöhung als natürlicher Treibhauseffekt bezeichnet. Der natürliche Treibhauseffekt führt dazu, dass die Durchschnittstemperatur der Erde bei etwa plus 14 Grad Celius liegt. Ohne den natürlichen Treibhauseffekt läge sie bei etwa minus 18 Grad Celius. Hierbei handelt es sich um rechnerisch bestimmte Werte. In der Literatur können diese Werte gegebenenfalls leicht abweichen, je nach Rechenansatz und der zu Grunde gelegten Annahmen, zum Beispiel dem Reflexionsverhalten der Erde. Diese Werte dienen als Nachweis, dass es einen natürlichen Treibhauseffekt gibt, da ohne ihn die Temperatur entsprechend deutlich geringer sein müsste und sich die höhere Temperatur mit dem Treibhauseffekt erklären lässt. Abweichungen von wenigen Grad Celsius spielen bei diesem Nachweis zunächst keine wesentliche Rolle.'''

segment_nlp = spacy.load('de')

# find parent in spacy dependency graph
def find_node(common_parent, search_node):
    path_len = 0
    while search_node != common_parent:
        path_len += 1
        search_node = search_node.head
    return path_len

# Segments the given text

# Options:
# beam_size: what beam size to use for the segmentation search
# ideal_token_len: the ideal length of tokens per segment
# len_reward_factor: how important it is to be close to ideal_token_len,
#                    higher factor = splits are closer to ideal_token_len
# sentence_end_reward_factor: the weight of the sentence end score in the search.
#                             Heigher values make it more likely to always split at sentence end.
# boost_comma_factor: boosts the probability to split at a comma (',')
# max_lookahead: maximum lookahead for the beam search, this is also the maximum length of one segment
# debug_print: print additional debug info

# Todo: allow other languages than German
def segment_beamsearch(text, beam_size=10, ideal_token_len=12, len_reward_factor=2.5,
                   sentence_end_reward_factor=0.8, boost_comma_factor=2.0, max_lookahead=40, debug_print=False):
    doc = segment_nlp(text)
    doc_parsetree_seqs = []    

    # Iterate over all sentences in the text
    for sent in doc.sents:
        if debug_print:
            print( [sent.start, sent.end] )
        span = doc[sent.start: sent.end]

        # This generates the lowest common ancestry matrix for the dependency tree of the current sentence
        lca = span.get_lca_matrix()

        row_size = len(lca[0])
        parsetree_seq = []

        for i in range(row_size-1):
            # For all words, we use the lowest common ancestry (LCA) matrix
            # to determine the shortest path between all adjacent words
            # with LCA we just need to lookup the path to the common ancestor for the word i and i+1
            common_parent = lca[i][i+1]
            cur_node = span[i]
            path_len = find_node(span[common_parent], cur_node)

            cur_node = span[i+1]
            path_len += find_node(span[common_parent], cur_node)
    
            parsetree_seq.append(path_len)
        # This is the score/reward for sentence end, we make it depended on the sentence length
        parsetree_seq.append(sentence_end_reward_factor * row_size)

        # We boost comma rewards, since they are usually a better place to split then what the shortest path suggests
        for i,token in enumerate(span):
            if token.text == ',':
                parsetree_seq[i] *= boost_comma_factor

        if debug_print:
            print('next word reward function:', list(zip([token for token in span],parsetree_seq)))
            #print(row_size)
            #print(span.text)
        doc_parsetree_seqs += parsetree_seq

    if debug_print:
        print(doc_parsetree_seqs)

    num_doc_tokens = len(doc_parsetree_seqs)
    assert(num_doc_tokens == len(doc))


    # Beam search for a solution until there is no improvement
    # This is classical beam search, for all candidates on the beam we expand
    # and calculate the score for splitting at any position between +1 and
    # we use the scores from doc_parsetree_seqs, longer shortest path = better point to split.
    # So bigger number = better.
    # Additionally we compute the difference to the ideal sequence length (in words currently) and add this to the score:
    # (ideal length - |(ideal length - length)|)

    # sequences are of this shape; first list keeps track of the split positions, seconds lists contains the spans
    # the float value is the combined score for the complete path
    sequences = [[list(), list(), 0.0]]

    cont_search = True
    while cont_search:
        all_candidates = sequences
        cont_search = False
        # Expand each current candidate
        for i in range(len(sequences)):
            seq_pos, spans, current_score = sequences[i]
            last_cut = (seq_pos[-1] if (len(seq_pos) > 0) else 0)
            score_at_k = sequences[-1][2]
            # search over all tokens, 1 to max_lookahead
            for j in range(1, min(max_lookahead, num_doc_tokens - last_cut -1)):
                len_reward = len_reward_factor * (ideal_token_len - math.fabs(ideal_token_len - float(j)))
                
                new_score = current_score + len_reward + doc_parsetree_seqs[last_cut + j]
                candidate = [seq_pos + [last_cut + j +1], spans + [doc[last_cut:last_cut + j +1]], new_score]
                all_candidates.append(candidate)

                # only continue the search, of atleast one of the candidates was better than the current score at k
                if new_score > score_at_k:
                    cont_search = True

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[2], reverse=True)
        # select k best
        sequences = ordered[:beam_size]

    if debug_print:
        for sequence in sequences:
            print(sequence[2])
  
    best = sequences[0]

    if debug_print:
        for span in best[1]:
            print(span.text)

    return [sp.text for sp in best[1]]

if __name__ == "__main__":
    print('test input')
    print(test)

    print()
    print('segments:')
    for segment in segment_beamsearch(test):
        print(segment)

