#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterRecognizer, LatticeLmRescorer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
import yaml

model_dir = "models/de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst/"

# yaml #TODO use YAML File
config_file = "models/kaldi_tuda_de_nnet3_chain2.yaml"
with open(config_file, 'r') as stream:
    model_yaml = yaml.safe_load(stream)
decoder_yaml_opts = model_yaml['decoder']
# print (decoder_yaml_opts)


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
    model_dir + "final.mdl", model_dir + "HCLG.fst", model_dir + "words.txt",
    decoder_opts=decoder_opts, decodable_opts=decodable_opts)


# Construct symbol table
symbols = SymbolTable.read_text(model_dir + "words.txt")
phi_label = symbols.find_index("#0")

# # Construct LM rescorer
# rescorer = LatticeLmRescorer.from_files("G.fst", "G.rescore.fst", phi_label)

# Define feature pipelines as Kaldi rspecifiers
feats_rspec = "ark:compute-mfcc-feats --config=%sconf/mfcc_hires.conf scp:wav.scp ark:- |" % model_dir
ivectors_rspec = (
    "ark:compute-mfcc-feats --config=%sconf/mfcc_hires.conf scp:wav.scp ark:-"
    " | ivector-extract-online2 --config=%sivector_extractor/ivector_extractor.conf ark:spk2utt ark:- ark:- |" % (model_dir, model_dir)
    )
# Decode wav files
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i:
    for (fkey, feats), (ikey, ivectors) in zip(f, i):
        assert(fkey == ikey)
        out = asr.decode((feats, ivectors))
        words, _, _ = get_linear_symbol_sequence(shortestpath(out["lattice"]))
        print(fkey, " ".join(indices_to_symbols(symbols, words)), flush=True)