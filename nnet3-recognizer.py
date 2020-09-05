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

models_dir = "models/"

# read yaml File
config_file = "models/kaldi_tuda_de_nnet3_chain2.yaml"
with open(config_file, 'r') as stream:
    model_yaml = yaml.safe_load(stream)
decoder_yaml_opts = model_yaml['decoder']


# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
# hier yaml Datei einfügen
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
        BP = functions.compact_lattice_shortest_path(out["lattice"])
        words, _, _ = get_linear_symbol_sequence(shortestpath(BP))
        # print(functions.compact_lattice_shortest_path(out["lattice"]))
        Timing = functions.compact_lattice_to_word_alignment(BP)
        # print(fkey, " ".join(indices_to_symbols(symbols, words)), flush=True)

Test = indices_to_symbols(symbols, Timing[0]) # Wandelt die Word Nummern um zu Wörtern
VTT = zip(Test, Timing[1], Timing[2]) # Erstellt Datenstruktur (Wort, Wortanfang (Frames), Wortende(Frames))
VTT = list(VTT)
print(VTT)
sequences = [["" for x in range(3)] for y in range(30)] # TODO: Variable Größe statt fester Arraygröße

def ArrayToSequences():
    wcounter = 0
    scounter = 0
    for a in VTT:
        if wcounter < 10:
            if wcounter == 0: # erstes Wort in der Sequenz
                sequences[scounter][1] = a[1] # Setzt Anfangstiming der Sequenz
            sequences[scounter][0] = sequences[scounter][0] + " " + (a[0]) # TODO: Vor jeder Sequenz ist immer ein überschüssiges Leerzeichen
            wcounter += 1
            sequences[scounter][2] = a[1] + a[2] # Setzt Endtiming der Sequenz
        else:
            wcounter = 1
            scounter += 1
            sequences[scounter][0] = sequences[scounter][0] + " " + (a[0])
            sequences[scounter][1] = a[1]
            sequences[scounter][2] = a[1] + a[2]            
    # print(sequences)

def createVTT():
    file = open("subtitle.vtt", "w") # TODO: In abhängigkeit zu Wave Datei / etc benennen
    file.write("WEBVTT\n")
    file.write("\n")
    sequenz_counter = 1
    for a in sequences:
        start_seconds = int(a[1] / 32) # Start der Sequenz in Sekunden TODO: Framerate bestimmen
        end_seconds = int(a[2] / 32) # Ende der Sequenz in Sekunden
        file.write(str(sequenz_counter) + "\n") # Nummer der aktuellen Sequenz TODO: Direkt in die Datenstruktur sequences einpflegen
        timestring = "00:" + str(int(start_seconds / 60)) + ":" + str(start_seconds % 60) + ".000 --> " + "00:" + str(int(end_seconds / 60)) + ":" + str((end_seconds % 60)) + ".000" + "\n" # Generiert 00:00:000 --> 00:00:000 TODO: Noch nicht nach Standard
        file.write(timestring)
        file.write(a[0] + "\n")
        file.write("\n")
        sequenz_counter += 1
    file.close()

ArrayToSequences()
createVTT()