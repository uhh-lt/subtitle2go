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

import argparse
import scipy
from scipy.io import wavfile
import ffmpeg
import pylab as plt
import math
from python_speech_features import logfbank
from scipy.ndimage.filters import gaussian_filter1d


# All timing are in frames, where one frame is 0.01 seconds.
def process_wav(wav_filename, beam_size=10, ideal_segment_len=100*60,
                max_lookahead=100*180, min_len=100*5, step=1, debug=False):

    samplerate, data = wavfile.read(wav_filename, mmap=False)
    fbank_feat = logfbank(data, samplerate=samplerate, winlen=0.025, winstep=0.01)
    fbank_feat_power = fbank_feat.sum(axis=-1) / 10.

    print(fbank_feat_power)

    fbank_feat_len = len(fbank_feat)

    fbank_feat_min_power = min(fbank_feat_power)
    fbank_feat_max_power = max(fbank_feat_power)

    fbank_feat_power_smoothed = gaussian_filter1d(fbank_feat_power, sigma=20) * -1.0

    if debug:
        print('min:', fbank_feat_min_power, 'max:', fbank_feat_max_power)

    if debug:
        plt.imshow(fbank_feat[:1000].T, interpolation=None, aspect='auto', origin='lower')
        plt.show()
        plt.plot(fbank_feat_power_smoothed[:1000])
        plt.show()

    cont_search = True

    len_reward_factor = 30. / float(ideal_segment_len)

    # Simple Beam search to find good cuts, where the eneregy is low and where its
    # still close to the ideal segment length.
    # sequences are of this shape; first list keeps track of the split positions,
    # the float value is the combined score for the complete path.
    sequences = [[[0], 0.0]]
    sequences_ordered = [[]]

    while cont_search:
        all_candidates = sequences
        cont_search = False
        # Expand each current candidate
        for i in range(len(sequences)):
            seq_pos, current_score = sequences[i]
            last_cut = (seq_pos[-1] if (len(seq_pos) > 0) else 0)
            score_at_k = sequences[-1][1]
            # search over all tokens, min_len to max_lookahead
            for j in range(min_len, min(max_lookahead, fbank_feat_len - last_cut - 1), step):
                len_reward = len_reward_factor * (ideal_segment_len - math.fabs(ideal_segment_len - float(j)))
                fbank_score = fbank_feat_power_smoothed[last_cut+j]
                new_score = current_score + len_reward + fbank_score
                if new_score > current_score:
                    #print("fbank_score:", fbank_score, "len reward:", len_reward)
                    candidate = [seq_pos + [last_cut + j + 1], new_score]
                    all_candidates.append(candidate)

                # only continue the search, of at least one of the candidates was better than the current score at k
                if new_score > score_at_k:
                    cont_search = True

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # select k best
        sequences_ordered = ordered[:beam_size]

    print(sequences_ordered)

    # this can happen with very short input wavs
    if len(sequences_ordered[0]) <= 1:
        segments = [(0, fbank_feat_len)]
    else:
        best_cuts = sequences_ordered[0]
        segments = list(zip(best_cuts[0][:-1], best_cuts[0][1:]))

    print('segments:', segments)

    if debug:
        for i, segment in enumerate(segments):
            print(segment)
            out_filename = "segments/%d.wav"%i
            print('Writing to:', out_filename)
            print('Segment len:', segment[1]-segment[0])
            wavfile.write(out_filename, samplerate, data[segment[0]*160:segment[1]*160])

    return segments


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='This tool does a simple endpointing beam search over a long audio'
                                                 ' file, to cut it into smaller pieces for ASR processing.')

    parser.add_argument('-a', '--average-segment-length', help='Average segment length in seconds.',
                                     type=float, default=60.0)

    # positional argument, without (- and --)
    parser.add_argument('filename', help='The path of the mediafile', type=str)

    args = parser.parse_args()
    filenameS = args.filename.rpartition('.')[0] # Filename without file extension
    filename = args.filename

    filenameS_hash = hex(abs(hash(filenameS)))[2:]

    tmp_file = 'tmp/%s.wav' % filenameS_hash

    # use ffmpeg to convert the input media file (any format!) to 16 kHz wav mono
    (
        ffmpeg
            .input(filename)
            .output(tmp_file, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run()
    )

    process_wav(tmp_file, debug=True)
