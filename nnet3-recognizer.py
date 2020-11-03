#!/usr/bin/env python
from __future__ import print_function

from kaldi.asr import NnetLatticeFasterRecognizer, LatticeLmRescorer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.lat import functions
import yaml
import math
import argparse
import ffmpeg
import os
import segment_text

# TODO: Umgebungsvariablen setzen oder Kaldi binaries in virtualenv verschieben
# os.environ["KALDI_ROOT"] = "pykaldi/tools/kaldi"
# os.environ["PATH"] = "pykaldi/tools/kaldi/src/lmbin/:pykaldi/tools/kaldi/../kaldi_lm/:" + os.getcwd() + "/utils/:pykaldi/tools/kaldi/src/bin:pykaldi/tools/kaldi/tools/openfst/bin:pykaldi/tools/kaldi/src/fstbin/:pykaldi/tools/kaldi/src/gmmbin/:pykaldi/tools/kaldi/src/featbin/:pykaldi/tools/kaldi/src/lm/:pykaldi/tools/kaldi/src/sgmmbin/:pykaldi/tools/kaldi/src/sgmm2bin/:pykaldi/tools/kaldi/src/fgmmbin/:pykaldi/tools/kaldi/src/latbin/:pykaldi/tools/kaldi/src/nnetbin:pykaldi/tools/kaldi/src/nnet2bin/:pykaldi/tools/kaldi/src/online2bin/:pykaldi/tools/kaldi/src/ivectorbin/:pykaldi/tools/kaldi/src/kwsbin:pykaldi/tools/kaldi/src/nnet3bin:pykaldi/tools/kaldi/src/chainbin:pykaldi/tools/kaldi/tools/sph2pipe_v2.5/:pykaldi/tools/kaldi/src/rnnlmbin:$PWD:$PATH"
# os.environ["PATH"] = "$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH"

def asr():
    models_dir = "models/"

    # Read yaml File
    config_file = "models/kaldi_tuda_de_nnet3_chain2.yaml"
    with open(config_file, 'r') as stream:
        model_yaml = yaml.safe_load(stream)
    decoder_yaml_opts = model_yaml['decoder']

    # ffmpeg
    (
        ffmpeg
        .input(filename)
        .output("new.wav", acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run()
    )

    # Construct recognizer
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 13
    decoder_opts.max_active = 7000
    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 3
    decodable_opts.frames_per_chunk = 150
    asr = NnetLatticeFasterRecognizer.from_files(
        models_dir + decoder_yaml_opts["model"], models_dir + decoder_yaml_opts["fst"], models_dir + decoder_yaml_opts["word-syms"],
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

    # Construct symbol table
    symbols = SymbolTable.read_text(models_dir + decoder_yaml_opts["word-syms"])
    phi_label = symbols.find_index("#0")

    # Define feature pipelines as Kaldi rspecifiers
    feats_rspec = "ark:compute-mfcc-feats --config=%s scp:wav.scp ark:- |" % (models_dir + decoder_yaml_opts["mfcc-config"])
    ivectors_rspec = (
        "ark:compute-mfcc-feats --config=%s scp:wav.scp ark:-"
        " | ivector-extract-online2 --config=%s ark:spk2utt ark:- ark:- |" % ((models_dir + decoder_yaml_opts["mfcc-config"]), (models_dir + decoder_yaml_opts["ivector-extraction-config"]))
        )

    # Decode wav files
    with SequentialMatrixReader(feats_rspec) as f, \
        SequentialMatrixReader(ivectors_rspec) as i:
        for (fkey, feats), (ikey, ivectors) in zip(f, i):
            assert(fkey == ikey)
            out = asr.decode((feats, ivectors))
            best_path = functions.compact_lattice_shortest_path(out["lattice"])
            words, _, _ = get_linear_symbol_sequence(shortestpath(best_path))
            # print(functions.compact_lattice_shortest_path(out["lattice"]))
            timing = functions.compact_lattice_to_word_alignment(best_path)
            # print(fkey, " ".join(indices_to_symbols(symbols, words)), flush=True)

    words = indices_to_symbols(symbols, timing[0]) # Wandelt die Word Nummern um zu Wörtern
    vtt = list(map(list, zip(words, timing[1], timing[2]))) # Erstellt Datenstruktur (Wort, Wortanfang(Frames), Wortende(Frames))
    return vtt, words


def array_to_squences(vtt): # Alte Sequenztrennung nach 10 Wörtern
    len_array = math.ceil(len(vtt) / 10)
    sequences = [["" for x in range(3)] for y in range(len_array)]
    wcounter = 0
    scounter = 0
    for a in vtt:
        if wcounter < 10:
            if wcounter == 0: # erstes Wort in der Sequenz
                sequences[scounter][1] = a[1] # Setzt Anfangstiming der Sequenz
                sequences[scounter][0] = a[0]
            else:
                sequences[scounter][0] = sequences[scounter][0] + " " + a[0]
            wcounter += 1
            sequences[scounter][2] = a[1] + a[2] # Setzt Endtiming der Sequenz
        else:
            wcounter = 1
            scounter += 1
            sequences[scounter][0] = a[0]
            sequences[scounter][1] = a[1]
            sequences[scounter][2] = a[1] + a[2]
    return sequences


def add_interpunctuation(vtt, words):
    print("Starting Interpunction")
    raw_file = open("raw_text.txt", "w")
    raw_file.write(' '.join(words))
    raw_file.close() # Schreibt die ASR Daten zu einer neuen Datei
    os.system("./punctuator.sh") # Startet Punctuator2 extern (fügt Interpunktion hinzu und macht den Text lesbar)
    file_punct = open("punc_output_readable.txt", "r") # liest den interpunktierten Text ein
    punct_list = file_punct.read().split(" ")
    vtt_punc = []
    for a,b in zip(punct_list, vtt): # Ersetzt die veränderten Wörter (Großschreibung, Punkt, Komma) mit den Neuen
        if a != b[0]:
            vtt_punc.append([a, b[1], b[2]])
        else:
            vtt_punc.append(b)
    return vtt_punc


def segmentation(vtt):
    sequences = []

    word_string = ""
    word_counter = 0
    print("Begin Segmentation")
    for e in vtt:
        if e[0] == "<UNK>": # Die <UNK> Token werden in der Segmentierung manchmal getrennt.
            word_string = word_string + " " + "UNK"
        else:
            word_string = word_string + " " + e[0]
    segments = segment_text.segment_beamsearch(word_string) # Die Segmentierung
    
    temp_segments = []
    temp_segments.append(segments[0])
    for current in segments[1:]: # Korrektur falls ein Satzzeichen in die nächste Zeile gerutscht ist
        if current[0] == "," or current[0] == ".":
            temp_segments[:1][0] = temp_segments[:1][0] + current[0]
            current = current[2:]
        temp_segments.append(current)
    segments = temp_segments

    for segment in segments:
        cleanSegment = list(filter(None, segment.split(" "))) # Trennt das Segment in einzelne Wörter, entfernt leere Objekte vom String und speichert eine Liste
        stringSegment = " ".join(cleanSegment) # Baut einen neuen String
        seg_len = len(cleanSegment) # Länge des aktuellen Segments
        if word_counter != 0: # Sonst überschneiden sich die Untertitel
            begin_segment = vtt[word_counter + 1][1]
        else:
            begin_segment = vtt[word_counter][1]
        end_segment =  vtt[word_counter + seg_len][1] + vtt[word_counter + seg_len][2]
        sequences.append([stringSegment, begin_segment, end_segment])
        word_counter = word_counter + seg_len
    return sequences


def create_subtitle(sequences, subtitleFormat):
    if subtitleFormat == "vtt":
        file = open(filenameS + ".vtt", "w")
        file.write("WEBVTT\n\n")
        separator = "."
    elif subtitleFormat == "srt":
        file = open(filenameS + ".srt", "w")
        separator = ","

    sequenz_counter = 1
    for a in sequences:
        start_seconds = a[1] / 33.333 # Start der Sequenz in Sekunden 
        end_seconds = a[2] / 33.333 # Ende der Sequenz in Sekunden
        file.write(str(sequenz_counter) + "\n") # Nummer der aktuellen Sequenz TODO: Direkt in die Datenstruktur sequences einpflegen
        if start_seconds == 0: # Erste Sequenz darf nicht bei 0 starten sonst wird sie nicht verarbeitet
            time_start = "00:00:00{}001" .format(separator)
        else:
            time_start = "{:0>2d}:{:0>2d}:{:0>2d}{}000".format(int(start_seconds / 3600), int((start_seconds / 60) % 60), int(start_seconds % 60), separator)
        time_end = "{:0>2d}:{:0>2d}:{:0>2d}{}000".format(int(end_seconds / 3600), int((end_seconds / 60) % 60), int(end_seconds % 60), separator)
        timestring = time_start + " --> " + time_end + "\n"
        file.write(timestring)
        file.write(a[0] + "\n\n")
        sequenz_counter += 1
    file.close()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="The path of the mediafile", type=str, required=True)
    parser.add_argument("-s", "--subtitle", help="The output subtitleformat (vtt or srt). Default=vtt", required=False, default="vtt", choices=["vtt", "srt"])
    args = parser.parse_args()
    filenameS = args.filename.rpartition(".")[0]
    filename = args.filename
    subtitleFormat = args.subtitle

    vtt, words = asr()
    vtt = add_interpunctuation(vtt, words)
    sequences = segmentation(vtt)
    # sequences = array_to_sequences(vtt)
    create_subtitle(sequences, subtitleFormat)