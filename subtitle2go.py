#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright 2022 HITeC e.V.
#
#    Licensed under the Apache License, Version 2.0 (the 'License');
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an 'AS IS' BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

# Load Kaldi
from kaldi.asr import NnetLatticeFasterRecognizer, LatticeLmRescorer, LatticeRnnlmPrunedRescorer
from kaldi.rnnlm import RnnlmComputeStateComputationOptions
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.lat.functions import ComposeLatticePrunedOptions
from kaldi.lat.align import read_lexicon_for_word_align
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.lat import functions
from kaldi.transform import cmvn

# Audio Segmentation
from simple_endpointing import process_wav

# Interpunctuation
from rpunct import RestorePuncts

import yaml
import argparse
import ffmpeg
import segment_text
import slide_stripper
import json
import time
import sys

start_time = time.time()

kaldi_feature_factor = 3.00151874884282680911

class output_status():
    def __init__(self, filename, fn_short_hash, redis=False):
        if redis:
            try:
                import redis
                self.red = redis.StrictRedis(charset='utf-8', decode_responses=True)
            except ImportError:
                print('Redis is not available. Disabling redis option.')
                redis = False

            self.redis_server_channel = 'subtitle2go'
        self.redis = redis

        self.filename = filename
        self.fn_short_hash = fn_short_hash

    def publish_status(self, status):
        print(f'{filename=} {self.fn_short_hash=} {status=}')
        if self.redis:
            self.red.publish(self.redis_server_channel, json.dumps({'pid': os.getpid(), 'time': time.time(), 'start_time': start_time,
                                                    'file_id': self.fn_short_hash, 'filename': self.filename,
                                                    'status': status}))

# Make sure a fpath directory exists
def ensure_dir(fpath):
    directory = os.path.dirname(fpath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_audio(filename, wav_filename):
    # Use ffmpeg to convert the input media file (any format!) to 16 kHz wav mono
    (
        ffmpeg
            .input(filename)
            .output(wav_filename, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
    )

def recognizer(decoder_yaml_opts, models_dir):
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = decoder_yaml_opts['beam']
    decoder_opts.max_active = decoder_yaml_opts['max-active']
    decoder_opts.lattice_beam = decoder_yaml_opts['lattice-beam']

    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = decoder_yaml_opts['acoustic-scale']
    decodable_opts.frame_subsampling_factor = 3 # decoder_yaml_opts['frame-subsampling-factor'] # 3
    decodable_opts.frames_per_chunk = 150
    fr = NnetLatticeFasterRecognizer.from_files(
        models_dir + decoder_yaml_opts['model'],
        models_dir + decoder_yaml_opts['fst'],
        models_dir + decoder_yaml_opts['word-syms'],
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

    return fr

# This method contains all Kaldi related calls and methods
def Kaldi(config_file, scp_filename, spk2utt_filename, segments_filename, do_rnn_rescore, segments_timing, lm_scale, acoustic_scale):

    models_dir = 'models/'

    # Read yaml File
    with open(config_file, 'r') as stream:
        model_yaml = yaml.safe_load(stream)
    decoder_yaml_opts = model_yaml['decoder']

    # Construct recognizer
    fr = recognizer(decoder_yaml_opts, models_dir)

    # Check if cmvn is set
    cmvn_transformer = None
    if decoder_yaml_opts.get('global-cmvn-stats'):
        cmvn_transformer = cmvn.Cmvn(40)
        cmvn_transformer.read_stats(f'{models_dir}{decoder_yaml_opts["global-cmvn-stats"]}')

    # Construct symbol table
    symbols = SymbolTable.read_text(models_dir + decoder_yaml_opts['word-syms'])
    # phi_label = symbols.find_index('#0')
#    segments_filename = f'{scp_filename.partition(".scp")[0]}_segments'
    # Define feature pipelines as Kaldi rspecifiers
    # feats_rspec = (f'ark:compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} scp:{scp_filename} ark:- |')
    feats_rspec = (f'ark:extract-segments scp,p:{scp_filename} {segments_filename} ark:- | compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} ark:- ark:- |')
    # ivectors_rspec = (
    #         (f'ark:compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} '
    #         f'scp:{scp_filename} ark:- | '
    #         f'ivector-extract-online2 --config={models_dir}{decoder_yaml_opts["ivector-extraction-config"]} '
    #         f'ark:{spk2utt_filename} ark:- ark:- |'))
    
    ivectors_rspec = (
            (f'ark:extract-segments scp,p:{scp_filename} {segments_filename} ark:- | compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} '
            f'ark:- ark:- | '
            f'ivector-extract-online2 --config={models_dir}{decoder_yaml_opts["ivector-extraction-config"]} '
            f'ark:{spk2utt_filename} ark:- ark:- |'))
    
    
    rnn_rescore_available = 'rnnlm' in decoder_yaml_opts

    if do_rnn_rescore and not rnn_rescore_available:
        status.publish_status("Warning, disabling RNNLM rescoring since 'rnnlm' is not in the decoder options of the .yaml config.")

    if do_rnn_rescore and rnn_rescore_available:
        status.publish_status('Loading language model rescorer.')
        rnn_lm_folder = models_dir + decoder_yaml_opts['rnnlm'] 
        arpa_G = models_dir + decoder_yaml_opts['arpa'] 
        old_lm = models_dir + decoder_yaml_opts['fst'] 

        print(f'Loading RNNLM rescorer from:{rnn_lm_folder} with ARPA from:{arpa_G} FST:{old_lm}')
        # Construct RNNLM rescorer
        symbols = SymbolTable.read_text(rnn_lm_folder+'/config/words.txt')
        rnnlm_opts = RnnlmComputeStateComputationOptions()
        rnnlm_opts.bos_index = symbols.find_index('<s>')
        rnnlm_opts.eos_index = symbols.find_index('</s>')
        rnnlm_opts.brk_index = symbols.find_index('<brk>')
        compose_opts = ComposeLatticePrunedOptions()
        compose_opts.lattice_compose_beam = 6
        print(f'rnnlm-get-word-embedding {rnn_lm_folder}/word_feats.txt {rnn_lm_folder}/feat_embedding.final.mat -|')
        print(f'{rnn_lm_folder}/final.raw')
        rescorer = LatticeRnnlmPrunedRescorer.from_files(
            arpa_G,
            f'rnnlm-get-word-embedding {rnn_lm_folder}/word_feats.txt {rnn_lm_folder}/feat_embedding.final.mat -|',
            f'{rnn_lm_folder}/final.raw', lm_scale=lm_scale, acoustic_scale=acoustic_scale, max_ngram_order=4,
            use_const_arpa=True, opts=rnnlm_opts, compose_opts=compose_opts)

    did_decode = False
    decoding_results = []

    segmentcounter = 1
    with SequentialMatrixReader(feats_rspec) as f, \
            SequentialMatrixReader(ivectors_rspec) as i:
            for (fkey, feats), (ikey, ivectors) in zip(f, i):
                status.publish_status(f'Decoding segment {segmentcounter} of {len(segments_timing)}.')
                if cmvn_transformer:
                    cmvn_transformer.apply(feats)
                did_decode = True
                assert (fkey == ikey)
                out = fr.decode((feats, ivectors))
                if do_rnn_rescore:
                    lat = rescorer.rescore(out['lattice'])
                else:
                    lat = out['lattice']
                best_path = functions.compact_lattice_shortest_path(lat)
                print(f'{type(best_path)=}')
                print(f'{type(shortestpath(best_path))=}')
                words, _, _ = get_linear_symbol_sequence(shortestpath(best_path))
                timing = functions.compact_lattice_to_word_alignment(best_path)
                decoding_results.append((words, timing))
                segmentcounter+=1
    print(decoding_results)
    # Concatenating the results of the segments and adding an offset to the segments
    words = []
    timing = [[],[],[]]
    for result in decoding_results:
        words.extend(result[0])

    # for a, b in zip(segments_timing,decoding_results):
    #     print(f'{a=}')
    #     print(f'{b[1][1]=}')
    #     print(f'{b[1][2]=}')
    # TODO Linear lässt es sich nicht lösen. Mit jedem Segment muss der Offset erhöht werden
    # correction_factor = (3000 - (len(segments_filenames) * 10 / 33))
    for result, offset in zip(decoding_results, segments_timing):
        if result[1][1]:
            timing[0].extend(result[1][0])
            # start = map(lambda x: int(x + (offset[0] / kaldi_feature_factor)), result[1][1])
            start = [x + (offset[0] / kaldi_feature_factor) for x in result[1][1]]
            print('offset')
            kaldi_time_to_seconds(offset[0] / kaldi_feature_factor, seperator=".")
            print('start')
            kaldi_time_to_seconds(start[0], seperator=".")

            timing[1].extend(start)
            timing[2].extend(result[1][2])
    starting = 0
    temp_timing = [[], [], []]
    # for word, time, length in zip(timing[0], timing[1], timing[2]):
    #     temp_timing[0].append(word)
    #     temp_timing[1].append(starting)
    #     temp_timing[2].append(length)
    #     starting = starting + length
    # timing = temp_timing
    # bitchS = timing[1]
    # bitchL = timing[2]
    # for S, L, N in zip(bitchS, bitchL, bitchS[1:]):
    #     ressi = S+L-N
    #     if ressi != 0:
    #         print(ressi)

    # Maps words to the numbers
    words = indices_to_symbols(symbols, timing[0])

    # Creates the datastructure (Word, begin(Frames), end(Frames))
    assert(len(words) == len(timing[1]))
    assert(len(timing[1]) == len(timing[2]))
    vtt = list(map(list, zip(words, timing[1], timing[2])))

    if debug_word_timing:
        with open('debug_output.txt', 'w') as f:
            for element in vtt:
                f.write(f'{element[1]} {kaldi_time_to_seconds(element[1], ".")} {kaldi_time_to_seconds(element[1] + element[2], ".")} {element[2]} {element[0]}\n')

    return vtt, did_decode, words

# This is the asr function that converts the videofile, split the video into segments and decodes
def asr(filenameS_hash, filename, asr_beamsize=13, asr_max_active=8000, acoustic_scale=1.0, lm_scale=0.5,
         do_rnn_rescore=False, config_file='models/kaldi_tuda_de_nnet3_chain2_de_722k.yaml'):

    print(f"{filenameS_hash=}")

    scp_filename = f'tmp/{filenameS_hash}.scp'
    segments_filename = f'tmp/{filenameS_hash}_segments'
    wav_filename = f'tmp/{filenameS_hash}.wav'
    spk2utt_filename = f'tmp/{filenameS_hash}_spk2utt'

    # Audio extraction
    status.publish_status('Extract audio.')
    
    try:
        preprocess_audio(filename, wav_filename)
    except ffmpeg.Error as e:
        status.publish_status('Audio extraction failed.')
        status.publish_status(f'Complete Errormessage: {e.stderr}')
        sys.exit(-1)

    status.publish_status('Audio extracted.')

    # Segmentation
    status.publish_status('Audio segmentation.')

    try:
        segments_filenames, segments_timing = process_wav(wav_filename)
    except Exception as e:
        status.publish_status('Audio segmentation failed.')
        status.publish_status(f'Complete Errormessage: {e}')
        sys.exit(-1)

    print(f'{segments_filenames=}')
    print(f'{segments_timing=}')
    # Write scp and spk2utt file
    
    with open(scp_filename, 'w') as wavscp, open(spk2utt_filename, 'w') as spk2utt:
        # segmentFilename = wav_filename.rpartition('.')[0]
        wavscp.write(f'{filenameS_hash} {wav_filename}\n')

        for i in range(len(segments_timing)):
            count_str = "%.4d" % i    
            spk2utt.write(f'{filenameS_hash} {filenameS_hash}_{count_str}\n')
        print(f'Wrote {wavscp=} and {spk2utt=}') 
    
    # with open(scp_filename, 'w') as wavscp, open(spk2utt_filename, 'w') as spk2utt:
    #     for segment in segments_filenames:
    #         segmentFilename = segment.rpartition('.')[0]
    #         wavscp.write(f'{segmentFilename} {segment}\n')
    #         spk2utt.write(f'{filenameS_hash} {segmentFilename}\n')

    # Decode wav files
    status.publish_status('Start ASR.')
    vtt, did_decode, words = Kaldi(config_file, scp_filename, spk2utt_filename, segments_filename, do_rnn_rescore, segments_timing, lm_scale, acoustic_scale)
    if did_decode:
        status.publish_status('ASR finished.')
    else:
        status.publish_status('ASR error.')
        sys.exit(-1)

    # Cleanup tmp files
    # try:
    #     os.remove(scp_filename)
    #     os.remove(wav_filename)
    #     os.remove(spk2utt_filename)
    #     for segment_file in segments_filenames:
    #         os.remove(segment_file)
    #     status.publish_status(f'files removed:{scp_filename=}, {wav_filename=}, {spk2utt_filename=}, {segments_filenames=}')
    # except Exception as e:
    #     status.publish_status(f'Removing files failed')
    #     status.publish_status(f'Complete Errormessage: {e}')

    status.publish_status('VTT finished.')

    return vtt, words


# Adds interpunctuation to the Kaldi output
def interpunctuation(vtt, words, filenameS_hash, model_punctuation):

    status.publish_status('Starting interpunctuation.')

    # BERT
    text = str(' '.join(words))
    rpunct = RestorePuncts(model='models/interpunct_de_rpunct')
    
    punct = rpunct.punctuate(text)
    punct = punct.replace('.', '. ').replace(',', ', ').replace('!', '! ').replace('?', '? ')
    punct = punct.replace('  ', ' ')
    punct_list = punct.split(' ')
#    punct_list = words
    print(f'{len(punct_list)=}')
    print(f'{len(words)=}')
    print(f'{len(vtt)=}')

    # punct_list = file_punct.read().split(' ')
    vtt_punc = []
    for a, b in zip(punct_list, vtt):  # Replaces the adapted words with the (capitalization, period, comma) with the new ones
        vtt_punc.append([a, b[1], b[2]])
        print(f"{a=}, {b=}")

    status.publish_status('Adding interpunctuation finished.')

    return vtt_punc


def kaldi_time_to_seconds(value, seperator):
    time = value * kaldi_feature_factor / 100
    print(f'{time=}')
    time_start =    (f'{int(time / 3600):02}:'
                            f'{int(time / 60 % 60):02}:'
                            f'{int(time % 60):02}'
                            f'{seperator}'
                            f'{int(time * 1000 % 1000):03}')
    print(f'{time_start=}')
    return time_start

# This creates a segmentation for the subtitles and make sure it can still be mapped to the Kaldi tokenisation
def segmentation(vtt, model_spacy, beam_size, ideal_token_len, len_reward_factor, comma_end_reward_factor,
                 sentence_end_reward_factor):
    sequences = []

    status.publish_status('Start text segmentation.')

    # Array starts at zero
    word_counter = -1
    
    # Makes a string for segmentation and change the <UNK> and <unk> Token to UNK
    word_string = ' '.join([e[0].replace('<UNK>', 'UNK').replace('<unk>', 'UNK') for e in vtt])
    
    # Call the segmentation beamsearch
    segments = segment_text.segment_beamsearch(word_string, model_spacy, beam_size=beam_size, ideal_token_len=ideal_token_len,
                                               len_reward_factor=len_reward_factor,
                                               sentence_end_reward_factor=sentence_end_reward_factor,
                                               comma_end_reward_factor=comma_end_reward_factor)
    
    print(segments)

    temp_segments = []
    temp_segments.append(segments[0])


    # Corrects punctuation marks and also lost tokens when they are slipped
    # to the beginning of the next line
    for current in segments[1:]:
        currentL = current.split(' ')
        if any(token in currentL[0][0] for token in (',', '.', '?', '!', "'s", "n't", "'re")):
            temp_segments[-1] += currentL[0]
            currentL = currentL[1:]
        temp_segments.append(' '.join(currentL))
    segments = temp_segments


    # Cuts the segments in words, removes empty objects and
    # and creates the sequences object
    for segment in segments:
        clean_segment = list(filter(None, segment.split(' ')))
        string_segment = ' '.join(clean_segment)
        segment_length = len(clean_segment)
        # Fixes problems with the first token. The first token is everytime 0
        if vtt[word_counter + 1][1] == 0:
            begin_segment = vtt[word_counter + 2][1]
        else:
            begin_segment = vtt[word_counter + 1][1]
        end_segment = vtt[word_counter + segment_length][1] + vtt[word_counter + segment_length][2]
        sequences.append([string_segment, begin_segment, end_segment])
        word_counter = word_counter + segment_length
    
    status.publish_status('Text segmentation finished.')
    
    return sequences

# Creates the subtitle in the desired subtitleFormat and writes to filenameS (filename stripped) + subtitle suffix
def create_subtitle(sequences, subtitle_format, filenameS):
    status.publish_status('Start creating subtitle.')

    try:
        if subtitle_format == 'vtt':
            file = open(filenameS + '.vtt', 'w')
            file.write('WEBVTT\n\n')
            separator = '.'
        elif subtitle_format == 'srt':
            file = open(filenameS + '.srt', 'w')
            separator = ','

        sequence_counter = 1
        for a in sequences:
            time_start = kaldi_time_to_seconds(a[1], separator)
            time_end = kaldi_time_to_seconds(a[2], separator)

            file.write(str(sequence_counter) + '\n')  # number of actual sequence

            timestring = time_start + ' --> ' + time_end + '\n'
            file.write(timestring)
            file.write(a[0] + '\n\n')
            sequence_counter += 1
        file.close()

    except Exception as e:
        status.publish_status('Creating subtitle failed.')
        status.publish_status(f'Complete Errormessage: {e}')
        sys.exit(-1)

    status.publish_status('Finished creating subtitle.')

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    # Flag (- and --) arguments
    parser.add_argument('-s', '--subtitle', help='The output subtitleformat (vtt or srt). Default=vtt',
                        required=False, default='vtt', choices=['vtt', 'srt'])

    parser.add_argument('-l', '--language', help='Sets the language of the models', required=False, default='de')
    
    parser.add_argument('-p', '--pdf', help='Path to the slides (PDF).',
                        required=False)

    parser.add_argument('-m', '--model-yaml', help='Kaldi model used for decoding (yaml config).',
                                     type=str, default='models/kaldi_tuda_de_nnet3_chain2_de_683k.yaml')

    parser.add_argument('--rnn-rescore', help='Do RNNLM rescoring of the decoder output (experimental,'
                                              ' needs more testing).',
                        action='store_true', default=False)

    parser.add_argument('--acoustic-scale', help='ASR decoder option: This is a scale on the acoustic'
                                                 ' log-probabilities, and is a universally used kludge'
                                                 ' in HMM-GMM and HMM-DNN systems to account for the'
                                                 ' correlation between frames.',
                         type=float, default=1.0)

    parser.add_argument('--asr-beam-size', help='ASR decoder option: controls the beam size in the beam search.'
                                                ' This is a speed / accuracy tradeoff.',
                        type=int, default=13)

    parser.add_argument('--asr-max-active', help='ASR decoder option: controls the maximum number of states that '
                                                 'can be active at one time.',
                        type=int, default=16000)

    parser.add_argument('--segment-beam-size', help='What beam size to use for the segmentation search',
                        type=int, default=10)
    parser.add_argument('--ideal-token-len', help='The ideal length of tokens per segment',
                        type=int, default=10)

    parser.add_argument('--len-reward-factor', help='How important it is to be close to ideal_token_len,'
                                                    ' higher factor = splits are closer to ideal_token_len',
                        type=float, default=2.3)
    parser.add_argument('--sentence-end-reward_factor', help='The weight of the sentence end score in the search.'
                                                             ' Higher values make it more likely to always split '
                                                             'at sentence end.',
                        type=float, default=0.9)
    parser.add_argument('--comma-end-reward-factor', help='The weight of the comma end score in the search. '
                                                          'Higher values make it more likely to'
                                                          ' always split at commas.',
                        type=float, default=0.5)

    parser.add_argument('--with-redis-updates', help='Update a redis instance about the current progress.',
                        action='store_true', default=False)

    parser.add_argument('--debug', help='Output debug timing information', default=False)

    # Positional argument, without (- and --)
    parser.add_argument('filename', help='The path of the mediafile', type=str)

    args = parser.parse_args()
    filenameS = args.filename.rpartition('.')[0] # Filename without file extension
    filename = args.filename
    subtitle_format = args.subtitle
    pdf_path = args.pdf
    model_kaldi = args.model_yaml
    debug_word_timing = args.debug

    filenameS_hash = hex(abs(hash(filenameS)))[2:]
    ensure_dir('tmp/')

    # Init status class
    status = output_status(redis=args.with_redis_updates, filename=filename, fn_short_hash=filenameS_hash)

    # Language selection
    language = args.language
    with open('languages.yaml', 'r') as stream:
        language_yaml = yaml.safe_load(stream)
        if language_yaml.get(language, None):
            model_kaldi = language_yaml[language]['kaldi']
            model_punctuation = language_yaml[language]['punctuation']
            model_spacy = language_yaml[language]['spacy']

        else:
            print(f'language {language} is not set in languages.yaml . exiting.')
            sys.exit()

    if (pdf_path):
        slides = slide_stripper.convert_pdf(pdf_path)

    vtt, words = asr(filenameS_hash, filename=filename, asr_beamsize=args.asr_beam_size,
                     asr_max_active=args.asr_max_active, acoustic_scale=args.acoustic_scale,
                     do_rnn_rescore=args.rnn_rescore, config_file=model_kaldi)
    
    vtt = interpunctuation(vtt, words, filenameS_hash, model_punctuation)

    sequences = segmentation(vtt, model_spacy, beam_size=args.segment_beam_size, ideal_token_len=args.ideal_token_len,
                             len_reward_factor=args.len_reward_factor,
                             sentence_end_reward_factor=args.sentence_end_reward_factor,
                             comma_end_reward_factor=args.comma_end_reward_factor)

    create_subtitle(sequences, subtitle_format, filenameS)

    status.publish_status('Job finished successfully.')
