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

models_dir = "models/"
sequences = []
# TODO: Umgebungsvariablen setzen oder Kaldi binaries in virtualenv verschieben
# os.environ["KALDI_ROOT"] = "pykaldi/tools/kaldi"
# os.environ["PATH"] = "pykaldi/tools/kaldi/src/lmbin/:pykaldi/tools/kaldi/../kaldi_lm/:" + os.getcwd() + "/utils/:pykaldi/tools/kaldi/src/bin:pykaldi/tools/kaldi/tools/openfst/bin:pykaldi/tools/kaldi/src/fstbin/:pykaldi/tools/kaldi/src/gmmbin/:pykaldi/tools/kaldi/src/featbin/:pykaldi/tools/kaldi/src/lm/:pykaldi/tools/kaldi/src/sgmmbin/:pykaldi/tools/kaldi/src/sgmm2bin/:pykaldi/tools/kaldi/src/fgmmbin/:pykaldi/tools/kaldi/src/latbin/:pykaldi/tools/kaldi/src/nnetbin:pykaldi/tools/kaldi/src/nnet2bin/:pykaldi/tools/kaldi/src/online2bin/:pykaldi/tools/kaldi/src/ivectorbin/:pykaldi/tools/kaldi/src/kwsbin:pykaldi/tools/kaldi/src/nnet3bin:pykaldi/tools/kaldi/src/chainbin:pykaldi/tools/kaldi/tools/sph2pipe_v2.5/:pykaldi/tools/kaldi/src/rnnlmbin:$PWD:$PATH"
# os.environ["PATH"] = "$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH"

# Read yaml File
config_file = "models/kaldi_tuda_de_nnet3_chain2.yaml"
with open(config_file, 'r') as stream:
    model_yaml = yaml.safe_load(stream)
decoder_yaml_opts = model_yaml['decoder']

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="The path of the mediafile", type=str, required=True)
parser.add_argument("-s", "--subtitle", help="The output subtitleformat (vtt or srt). Default=vtt", required=False, default="vtt", choices=["vtt", "srt"])
parser.add_argument("-c", "--convert", help="Disable ffmpeg converter", required=False, action="store_false") # TODO: Wenn nicht konvertiert wird muss die wav.scp angepasst werden auf die neue Datei
args = parser.parse_args()
filenameS = args.filename.rpartition(".")[0]
filename = args.filename
subtitleFormat = args.subtitle
useFFMPEG = args.convert

# ffmpeg
if useFFMPEG:
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


# Decode wav files TODO: Gibt es eine Möglichkeit statt der wav.scp Datei direkt einen Stream an (py)kaldi zu übergeben um nicht erst die Datei auf die Festplatte schreiben zu müssen?
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i:
    for (fkey, feats), (ikey, ivectors) in zip(f, i):
        assert(fkey == ikey)
        out = asr.decode((feats, ivectors))
        BP = functions.compact_lattice_shortest_path(out["lattice"])
        words, _, _ = get_linear_symbol_sequence(shortestpath(BP))
        # print(functions.compact_lattice_shortest_path(out["lattice"]))
        Timing = functions.compact_lattice_to_word_alignment(BP)
        # print(fkey, " ".join(indices_to_symbols(symbols, words)), flush=True)

Words = indices_to_symbols(symbols, Timing[0]) # Wandelt die Word Nummern um zu Wörtern
VTT = zip(Words, Timing[1], Timing[2]) # Erstellt Datenstruktur (Wort, Wortanfang (Frames), Wortende(Frames))
VTT = list(VTT)
len_Array = math.ceil(len(VTT) / 10)


def ArrayToSequences(): # TODO: Überarbeiten wenn Sequenztrennung geklärt
    sequences = [["" for x in range(3)] for y in range(len_Array)]
    wcounter = 0
    scounter = 0
    for a in VTT:
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
    print(sequences)


def CreateSubtitle(subtitleFormat):
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

def AddInterpunctuation():
    global VTT
    print("Starting Interpunction")
    raw_file = open("raw_text.txt", "w")
    raw_file.write(' '.join(Words))
    raw_file.close() # Schreibt die ASR Daten zu einer neuen Datei
    os.system("./punctuator.sh") # Startet Punctuator2 extern (fügt Interpunktion hinzu und macht den Text lesbar)
    file_punct = open("punc_output_readable.txt", "r") # öffnet den interpunktierten Text
    punct_list = file_punct.read().split(" ")
    VTT_punc = []
    for a,b in zip(punct_list, VTT): # Ersetzt die veränderten Wörter (Großschreibung, Punkt, Komma) mit den neuen
        if a != b[0]:
            VTT_punc.append((a, b[1], b[2]))
        else:
            VTT_punc.append(b)
    VTT = VTT_punc

def Segmentierung():
    global sequences
    word_string = ""
    word_counter = 0
    print("Begin Segmentation")
    for e in VTT:
        word_string = word_string + " " + e[0]

    segments = segment_text.segment_beamsearch(word_string) # Die Segmentierung

    for segment in segments:
        cleanSegment = list(filter(None, segment.split(" "))) # Trennt das Segment in einzelne Wörter, entfernt leere Objekte vom String und speichert eine Liste
        stringSegment = " ".join(cleanSegment) # Baut einen neuen String
        seg_len = len(cleanSegment) # Länge des aktuellen Segments
        if word_counter != 0: # Sonst überschneiden sich die Untertitel
            begin_segment = VTT[word_counter+1][1]
        else:
            begin_segment = VTT[word_counter][1]
        end_segment =  VTT[word_counter + seg_len][1] + VTT[word_counter + seg_len][2]
        sequences.append([stringSegment, begin_segment, end_segment])
        word_counter = word_counter + seg_len


AddInterpunctuation()
Segmentierung()
# ArrayToSequences()
CreateSubtitle(subtitleFormat)