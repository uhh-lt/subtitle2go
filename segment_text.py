#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright 2022 HITeC e.V.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import spacy
import math

test = '''Seit der Industriellen Revolution verstärkt der Mensch den natürlichen Treibhauseffekt durch den Ausstoß von Treibhausgasen, wie messtechnisch belegt werden konnte. Seit 1990 ist der Strahlungsantrieb das heißt die Erwärmungswirkung auf das Klima durch langlebige Treibhausgase um 43 Prozent gestiegen. In der Klimatologie ist es heute Konsens, dass die gestiegene Konzentration der vom Menschen in die Erdatmosphäre freigesetzten Treibhausgase mit hoher Wahrscheinlichkeit die wichtigste Ursache der globalen Erwärmung ist, da ohne sie die gemessenen Temperaturen nicht zu erklären sind. Treibhausgase lassen die von der Sonne kommende kurzwellige Strahlung weitgehend ungehindert auf die Erde durch, absorbieren aber einen Großteil der von der Erde ausgestrahlten Infrarotstrahlung. Dadurch erwärmen sie sich und emittieren selbst Strahlung im langwelligen Bereich (vgl. Kirchhoffsches Strahlungsgesetz). Der in Richtung der Erdoberfläche gerichtete Strahlungsanteil wird als atmosphärische Gegenstrahlung bezeichnet. Im isotropen Fall wird die absorbierte Energie je zur Hälfte in Richtung Erde und Weltall abgestrahlt. Hierdurch erwärmt sich die Erdoberfläche stärker, als wenn allein die kurzwellige Strahlung der Sonne sie erwärmen würde. Das IPCC schätzt den Grad des wissenschaftlichen Verständnisses über die Wirkung von Treibhausgasen als hoch ein. Das Treibhausgas Wasserdampf trägt mit 36 bis 66 Prozent, Kohlenstoffdioxid mit 9 bis 26 Prozent und Methan mit 4 bis 9 Prozent zum natürlichen Treibhauseffekt bei. Die große Bandbreite erklärt sich folgendermaßen: Einerseits gibt es sowohl örtlich wie auch zeitlich große Schwankungen in der Konzentration dieser Gase. Zum anderen überlappen sich deren Absorptionsspektren. Beispiel: Strahlung, die von Wasserdampf bereits absorbiert wurde, kann von CO2 nicht mehr absorbiert werden. Das bedeutet, dass in einer Umgebung wie eisbedeckte Flächen oder Trockenwüste, in der Wasserdampf nur wenig zum Treibhauseffekt beiträgt, die übrigen Treibhausgase mehr zum Gesamttreibhauseffekt beitragen als in den feuchten Tropen. Da die genannten Treibhausgase natürliche Bestandteile der Atmosphäre sind, wird die von ihnen verursachte Temperaturerhöhung als natürlicher Treibhauseffekt bezeichnet. Der natürliche Treibhauseffekt führt dazu, dass die Durchschnittstemperatur der Erde bei etwa plus 14 Grad Celius liegt. Ohne den natürlichen Treibhauseffekt läge sie bei etwa minus 18 Grad Celius. Hierbei handelt es sich um rechnerisch bestimmte Werte. In der Literatur können diese Werte gegebenenfalls leicht abweichen, je nach Rechenansatz und der zu Grunde gelegten Annahmen, zum Beispiel dem Reflexionsverhalten der Erde. Diese Werte dienen als Nachweis, dass es einen natürlichen Treibhauseffekt gibt, da ohne ihn die Temperatur entsprechend deutlich geringer sein müsste und sich die höhere Temperatur mit dem Treibhauseffekt erklären lässt. Abweichungen von wenigen Grad Celsius spielen bei diesem Nachweis zunächst keine wesentliche Rolle.'''

test = '''In Husum Zeiten muss man ja immer pünktlich sein, was sonst akademischen werden. Nicht so richtig eher. Der Fall war das fing Fangnetz aber schon mal an. Ich freue mich sehr, dass wir so früh morgens am jetzt hier zusammenkommen. Einmal eben auf dem Campus von Melle Park. Und einmal in der Summe Welt. Also auf dem Server der Unität Hamburg-Mitte, das Jahr stattfinden Herzlich. Willkommen zu dieser Veranstaltung mit dem Titel Schulden Phobie und Lohnverzicht, wie man die Corona Krise zur Katastrophe macht. Mit er Professor Doktor Heiner Flassbeck er diese Veranstaltung findet, starb waren. Die meisten Menschen wissen, dass heute mal kurz er im Rahmen des ersten Semesters des Fachbereich Sozialökonomie. Das hat mal begonnen. Diese Idee eigentlich mit dem GzwanzigGipfel. Als hier in Steindorf entfernen, Messehallen sich die großen Welt Köpfe. Getroffen, haben wir gesagt da Moment mal, da haben wir noch einiges mehr mitzureden als Universität. Wollen uns mal diese Semester mit diesen Fragen auseinandersetzen und wollen ja, dass es im schüttel mich vor allem darauf ankommen könnte, Pons runter zu reißen. Oder für Lehrende. Dass man vor allem der pro Tag machen muss oder so. Sondern dass man sich mit den gesellschaftlich relevanten Fragen im im Studium beschäftigt. Dass es nicht so trocken schwimmen ist, sondern auch etwas für drittes relevantes gemeinsam machte, hatten dann auch schon zehn Semester zu Austritten aus Solidarität, Einzug, Gesundheitspflege und Kinderarbeit. Und haben jetzt eben uns in diesem mal überlegt, dass wir die gesellschaftliche Polarisierung sozialökonomische betrachten wollen. Also politisch, ökonomisch, kulturell, sozial, ökonomisch eben. Und haben uns dann aus aktuellem Anlass gesagt, dass man das auch in Zeiten von Corona eben zuspitzen muss, dass das jetzt sehr ansteht. Das Hochschulen sich da einmischen auch gegen diese ganze Erzählung, von dass eine Naturkatastrophe, was er die Hamburgische Bürgerschaft, die Masche beschlossen hat, um dann die Schuldenbremse Ausnahmeregelung anwenden zu können, um deutlich zu machen, dass das eine Gesellschaft Krise erheben is, wo wir eben auch dann handlungsfähig sind und uns das Jahr zu beschäftigen haben. Online Vorlesungen gemacht. Und sind jetzt eben diese Aktionswoche mittlerweile im finalen Tag am Freitag angekommen. Nachdem wir schon Hm, rechtswissenschaftliche, Medien, Soziologische, antirassistische sozialstaatliche, unser weite Diskussionen geführt haben, dann genau Sind wir jetzt eben er dabei und freuen uns sehr, dass wir das am ökonomisch diskutieren kann. Heterodoxe Ökonomie diskutieren können. Hm Genau mit eben Heiner Flassbeck, der beim Professor A an der ihm einen huschen Wirtschaftspolitik heutigen Fachbereich Sozialökonomie Immunität Hamburg is. Denn er war früher Stadtsekretär Bundesministerium. Der Finanzen sozusagen, hat dann langfristig rechtzeitig den Absprung gemacht, vor die neoliberale Phase dann eingeläutet is also sozusagen. Die Geschichte hat ihn daraufhin Fall recht gegeben. Was den offenen Krise und so alles folgte. Er war dann im Anschluss Chefökonom. Ähm, der UNO Organisationen für Welthandel und Entwicklungen unkt hat beim Hohen. Genau ist er seit zwei Tausend. Neunzehn Herausgeber er unter anderem dem fielen der Tätigkeiten, der Online-Zeitschrift makroskopisch Hm Genau Und. Ja, Mach schon länger im Lehrveranstaltungen. Vor allem in Master. Komische soziologische Studien. Aber Wirtschaftsgesellschaft ich auch ein bisschen Werbung machen will an dieser Stelle. Die Bewerbungsfrist läuft gerade. Also traut euch Ärmel. Genau. Und insofern freue ich mich jetzt, dass wir diese diese Diskussion Schuldenfalle Lohnverzicht, wie man die Corona Krise zu Katastrophe macht. Und vielleicht Alternativ oder so. Dann noch Gemeinde des gesunden. Ja auch normal. Dann besprechen wir das. Sind sie gemeinsam angehen kann ich wird jetzt auch gleich meine Klappe halten. Nur kurzer Hinweis auf Organisatorisches. Wenn ihr dann gleich nach dem Vortrag etwas sagen Wolfs in der Sagenwelt, dann wär 's cool, wenn ihr ein Ausrufezeichen in den Depots sind an sich, dass hier vor Ort und würdigsten sanken, die Rednerliste einsortieren. Und ich guck mich einfach hier bis in um auf informeller Park. Wer sich hier meldet. Und genau wurde das dann eben gemeinsam ein redete.'''

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
#                             Higher values make it more likely to always split at sentence end.
# comma_end_reward_factor: the weight of the comma end score in the search.
#                             Higher values make it more likely to always split at sentence end.
# max_lookahead: maximum lookahead for the beam search, this is also the maximum length of one segment
# debug_print: print additional debug info

def segment_beamsearch(text, model_spacy, beam_size=10, ideal_token_len=10, len_reward_factor=2.3,
                   sentence_end_reward_factor=0.9, comma_end_reward_factor=0.5, max_lookahead=40, debug_print=False):
    segment_nlp = spacy.load(model_spacy)
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
                #parsetree_seq[i] *= boost_comma_factor
                parsetree_seq[i] = comma_end_reward_factor * row_size

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
    segment_nlp = spacy.load('de_core_news_lg')
    print('test input')
    print(test)

    print()
    print('segments:')
    for segment in segment_beamsearch(test, segment_nlp):
        print(segment)
