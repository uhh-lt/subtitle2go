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
import slide_stripper

#make sure a fpath directory exists
def ensure_dir(fpath):
    directory = os.path.dirname(fpath)
    if not os.path.exists(directory):
        os.makedirs(directory)


# This is the main function that sets up the Kaldi decoder, loads the model and sets it up to decode the input file.
def asr(filenameS_hash, filenameS, asr_beamsize=13, asr_max_active=8000):
    models_dir = "models/"

    # Read yaml File
    config_file = "models/kaldi_tuda_de_nnet3_chain2.yaml"
    with open(config_file, 'r') as stream:
        model_yaml = yaml.safe_load(stream)
    decoder_yaml_opts = model_yaml['decoder']

    scp_filename = "tmp/%s.scp" % filenameS_hash
    wav_filename = "tmp/%s.wav" % filenameS_hash
    spk2utt_filename = "tmp/%s_spk2utt" % filenameS_hash

    # write scp file
    with open(scp_filename, 'w') as scp_file:
        scp_file.write("%s tmp/%s.wav\n" % (filenameS_hash, filenameS_hash))

    # write scp file
    with open(spk2utt_filename, 'w') as scp_file:
        scp_file.write("%s %s\n" % (filenameS_hash, filenameS_hash))

    # use ffmpeg to convert the input media file (any format!) to 16 kHz wav mono
    (
        ffmpeg
            .input(filename)
            .output("tmp/%s.wav" % filenameS_hash, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run()
    )

    # Construct recognizer
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = asr_beamsize
    decoder_opts.max_active = asr_max_active
    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 3
    decodable_opts.frames_per_chunk = 150
    asr = NnetLatticeFasterRecognizer.from_files(
        models_dir + decoder_yaml_opts["model"],
        models_dir + decoder_yaml_opts["fst"],
        models_dir + decoder_yaml_opts["word-syms"],
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

    # Construct symbol table
    symbols = SymbolTable.read_text(models_dir + decoder_yaml_opts["word-syms"])
    phi_label = symbols.find_index("#0")

    # Define feature pipelines as Kaldi rspecifiers
    feats_rspec = ("ark:compute-mfcc-feats --config=%s scp:" + scp_filename + " ark:- |") % \
                  (models_dir + decoder_yaml_opts["mfcc-config"])
    ivectors_rspec = (
            ("ark:compute-mfcc-feats --config=%s scp:" + scp_filename + " ark:-"
             + " | ivector-extract-online2 --config=%s ark:" + spk2utt_filename + " ark:- ark:- |") %
            ((models_dir + decoder_yaml_opts["mfcc-config"]),
             (models_dir + decoder_yaml_opts["ivector-extraction-config"]))
    )

    did_decode = False
    # Decode wav files
    with SequentialMatrixReader(feats_rspec) as f, \
            SequentialMatrixReader(ivectors_rspec) as i:
        for (fkey, feats), (ikey, ivectors) in zip(f, i):
            did_decode = True
            assert (fkey == ikey)
            out = asr.decode((feats, ivectors))
            best_path = functions.compact_lattice_shortest_path(out["lattice"])
            words, _, _ = get_linear_symbol_sequence(shortestpath(best_path))
            timing = functions.compact_lattice_to_word_alignment(best_path)

    assert(did_decode)

    # Maps words to the numbers
    words = indices_to_symbols(symbols, timing[0])

    # Creates the datastructure (Word, begin(Frames), end(Frames))
    vtt = list(map(list, zip(words, timing[1], timing[2])))

    # Cleanup tmp files
    print('removing tmp file:', scp_filename)
    os.remove(scp_filename)
    print('removing tmp file:', wav_filename)
    os.remove(wav_filename)
    print('removing tmp file:', spk2utt_filename)
    os.remove(spk2utt_filename)
    return vtt, words


def array_to_squences(vtt):  # Alte Sequenztrennung nach 10 Wörtern
    len_array = math.ceil(len(vtt) / 10)
    sequences = [["" for x in range(3)] for y in range(len_array)]
    wcounter = 0
    scounter = 0
    for a in vtt:
        if wcounter < 10:
            if wcounter == 0:  # erstes Wort in der Sequenz
                sequences[scounter][1] = a[1]  # Setzt Anfangstiming der Sequenz
                sequences[scounter][0] = a[0]
            else:
                sequences[scounter][0] = sequences[scounter][0] + " " + a[0]
            wcounter += 1
            sequences[scounter][2] = a[1] + a[2]  # Setzt Endtiming der Sequenz
        else:
            wcounter = 1
            scounter += 1
            sequences[scounter][0] = a[0]
            sequences[scounter][1] = a[1]
            sequences[scounter][2] = a[1] + a[2]
    return sequences


# Adds interpunctuation to the Kaldi output
def interpunctuation(vtt, words, filenameS_hash):
    raw_filename = "tmp/%s_raw.txt" % (filenameS_hash)
    token_filename = "tmp/%s_token.txt" % (filenameS_hash)
    readable_filename = "tmp/%s_readable.txt" % (filenameS_hash)
    
    print("Starting interpunctuation")
    
    raw_file = open(raw_filename, "w")
    raw_file.write(' '.join(words))
    raw_file.close()  # Schreibt die ASR Daten zu einer neuen Datei
    os.system("./punctuator.sh %s %s %s" % (raw_filename, token_filename, readable_filename))  # Starts Punctuator2 to add interpunctuation
    file_punct = open(readable_filename, "r")
    punct_list = file_punct.read().split(" ")
    vtt_punc = []
    for a, b in zip(punct_list, vtt):  # Ersetzt die veränderten Wörter (Großschreibung, Punkt, Komma) mit den Neuen
        if a != b[0]:
            vtt_punc.append([a, b[1], b[2]])
        else:
            vtt_punc.append(b)
    
    # Cleanup tmp files
    print('removing tmp file:', raw_filename)
    os.remove(raw_filename)
    print('removing tmp file:', token_filename)
    os.remove(token_filename)
    print('removing tmp file:', readable_filename)
    os.remove(readable_filename)

    return vtt_punc


# This creates a segmentation for the subtitles and make sure it can still be mapped to the Kaldi tokenisation
def segmentation(vtt, beam_size, ideal_token_len, len_reward_factor, comma_end_reward_factor,
                 sentence_end_reward_factor):
    sequences = []

    word_counter = -1 # array starts at zero
    print("Begin Segmentation")
    
    # Makes a string for segmentation and change the <UNK> Token to UNK
    word_string = " ".join([e[0].replace("<UNK>", "UNK") for e in vtt])
    
    # Call the segmentation beamsearch
    segments = segment_text.segment_beamsearch(word_string, beam_size=beam_size, ideal_token_len=ideal_token_len,
                                               len_reward_factor=len_reward_factor,
                                               sentence_end_reward_factor=sentence_end_reward_factor,
                                               comma_end_reward_factor=comma_end_reward_factor)
    
    temp_segments = []
    temp_segments.append(segments[0])
    # Corrects punctuation marks when they are slipped
    # to the beginning of the next line
    for current in segments[1:]:    
        if current[0] == "," or current[0] == ".":
            temp_segments[-1]+= current[0]
            current = current[2:]
        temp_segments.append(current)
    segments = temp_segments
    # Cuts the segments in words, removes empty objects and
    # and creates the sequences object
    for segment in segments:
        clean_segment = list(filter(None, segment.split(" ")))
        string_segment = " ".join(clean_segment)
        segment_length = len(clean_segment)
        # fixes problems with the first token. The first token is everytime 0.
        if vtt[word_counter + 1][1] == 0:
            begin_segment = vtt[word_counter + 2][1]
        else:
            begin_segment = vtt[word_counter + 1][1]
        end_segment = vtt[word_counter + segment_length][1] + vtt[word_counter + segment_length][2]
        sequences.append([string_segment, begin_segment, end_segment])
        word_counter = word_counter + segment_length
    return sequences


# Creates the subtitle in the desired subtitleFormat and writes to filenameS (filename stripped) + subtitle suffix
def create_subtitle(sequences, subtitle_format, filenameS):
    print("Creating subtitle")
    
    if subtitle_format == "vtt":
        file = open(filenameS + ".vtt", "w")
        file.write("WEBVTT\n\n")
        separator = "."
    elif subtitle_format == "srt":
        file = open(filenameS + ".srt", "w")
        separator = ","

    sequenz_counter = 1
    for a in sequences:

        start_seconds = a[1] / 33.333  # Start of sequence in seconds
        end_seconds = a[2] / 33.333  # End of sequence in seconds
        file.write(str(sequenz_counter) + "\n")  # number of actual sequence
        time_start = "{:0>2d}:{:0>2d}:{:0>2d}{}000".format(int(start_seconds / 3600),
                                                               int((start_seconds / 60) % 60),
                                                               int(start_seconds % 60), separator)

        time_end = "{:0>2d}:{:0>2d}:{:0>2d}{}000".format(int(end_seconds / 3600),
                                                         int((end_seconds / 60) % 60),
                                                         int(end_seconds % 60), separator)
        timestring = time_start + " --> " + time_end + "\n"
        file.write(timestring)
        file.write(a[0] + "\n\n")
        sequenz_counter += 1
    file.close()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()

    # flag (- and --) arguments
    parser.add_argument("-s", "--subtitle", help="The output subtitleformat (vtt or srt). Default=vtt",
                        required=False, default="vtt", choices=["vtt", "srt"])
    
    parser.add_argument("-p", "--pdf", help="Path to the slides (PDF).",
                        required=False)

    parser.add_argument("--asr-beam-size", help="ASR decoder option: controls the beam size in the beam search."
                                                " This is a speed / accuracy tradeoff.",
                        type=int, default=13)

    parser.add_argument("--asr-max-active", help="ASR decoder option: controls the maximum number of states that "
                                                 "can be active at one time.",
                        type=int, default=8000)

    parser.add_argument("--segment-beam-size", help="What beam size to use for the segmentation search",
                        type=int, default=10)
    parser.add_argument("--ideal-token-len", help="The ideal length of tokens per segment",
                        type=int, default=10)

    parser.add_argument("--len-reward-factor", help="How important it is to be close to ideal_token_len,"
                                                    " higher factor = splits are closer to ideal_token_len",
                        type=float, default=2.3)
    parser.add_argument("--sentence-end-reward_factor", help="The weight of the sentence end score in the search."
                                                             " Higher values make it more likely to always split "
                                                             "at sentence end.",
                        type=float, default=0.9)
    parser.add_argument("--comma-end-reward-factor", help="The weight of the comma end score in the search. "
                                                          "Higher values make it more likely to always split at commas.",
                        type=float, default=0.5)

    # positional argument, without (- and --)
    parser.add_argument("filename", help="The path of the mediafile", type=str)

    args = parser.parse_args()
    filenameS = args.filename.rpartition(".")[0] # Filename without file extension
    filename = args.filename
    subtitle_format = args.subtitle
    pdf_path = args.pdf

    filenameS_hash = hex(abs(hash(filenameS)))[2:]
    ensure_dir('tmp/')
    
    if (pdf_path):
        slides = slide_stripper.convert_pdf(pdf_path)

    vtt, words = asr(filenameS_hash, filenameS=filenameS, asr_beamsize=args.asr_beam_size, asr_max_active=args.asr_max_active)
    vtt = interpunctuation(vtt, words, filenameS_hash)
    sequences = segmentation(vtt, beam_size=args.segment_beam_size, ideal_token_len=args.ideal_token_len,
                             len_reward_factor=args.len_reward_factor,
                             sentence_end_reward_factor=args.sentence_end_reward_factor,
                             comma_end_reward_factor=args.comma_end_reward_factor)

    # sequences = array_to_sequences(vtt)
    create_subtitle(sequences, subtitle_format, filenameS)
