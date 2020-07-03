import os
import simpleaudio
import argparse
import nltk
from nltk.corpus import cmudict
import re
import numpy as np
from datetime import datetime

### NOTE: DO NOT CHANGE ANY OF THE EXISTING ARGUMENTS
parser = argparse.ArgumentParser(
    description='A basic text-to-speech app that synthesises an input phrase using diphone unit selection.')
parser.add_argument('--diphones', default="./diphones", help="Folder containing diphone wavs")
parser.add_argument('--play', '-p', action="store_true", default=False, help="Play the output audio")
parser.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                    default=None)
parser.add_argument('phrase', nargs=1, help="The phrase to be synthesised")

# Arguments for extensions
parser.add_argument('--spell', '-s', action="store_true", default=False,
                    help="Spell the phrase instead of pronouncing it")
parser.add_argument('--reverse', '-r', action="store_true", default=False,
                    help="Speak backwards")
parser.add_argument('--crossfade', '-c', action="store_true", default=False,
					help="Enable slightly smoother concatenation by cross-fading between diphone units")
parser.add_argument('--volume', '-v', default=None, type=int,
                    help="An int between 0 and 100 representing the desired volume")

args = parser.parse_args()
if args.volume is not None:
    if args.volume > 100:  # set volume to 100 if user input larger than 100
        print('the volume number needs to be between 0 and 100, for now it set to 100')
        args.volume = 100
    elif args.volume < 0:  # set volume to 0 if user input negative number
        print('the volume number needs to be between 0 and 100, for now it set to 0')
        args.volume = 0

def num_to_word(num):  # transfer a 0-99 number to English word
    lv1 = "zero one two three four five six seven eight nine ten \
           eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    lv2 = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    words = ''
    if int(num/10) >= 2:  # transfer the tens digit to English word
        words += lv2[int(num/10)-2]
        if num%10 > 0:
            words += ' '+lv1[num%10]
    else:
        words = lv1[num]  # transfer the unit digit to English word
    words = words.split()
    return words

date = ['zeroth','first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth', \
'eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth','nineteenth', \
'twentieth','twenty first','twenty second','twenty third','twenty fourth','twenty fifth','twenty sixth', \
'twenty seventh','twenty eighth','twenty ninth','thirtieth','thirty first']
# make up a date list for transfer date number to English word

class Synth:
    def __init__(self, wav_folder):
        self.diphones = {}
        self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        self.wavs = {}
        for root, dirs, files in os.walk(wav_folder, topdown=False):  # store all the diphone files into a dictionary
            for file in files:
                self.wavs[file[:-4].upper()] = os.path.join(root, file)

class Utterance:
    def __init__(self, phrase):
        if args.spell is True:
            pattern = r'[a-zA-Z]+'
            self.tokens = nltk.tokenize.regexp_tokenize(phrase.lower(), pattern)
            self.tokens = ''.join(self.tokens)  # transfer the words into letter sequence
        else:
            pattern = r'([a-zA-Z]+|,|[.:?!]|\d+\/\d+\/\d+|\d+\/\d+)'
            self.tokens = nltk.tokenize.regexp_tokenize(phrase.lower(), pattern)  # tokenize words date and punctuations

    def get_phone_seq(self):
        if args.spell is True:  # for spell
            seq = ['PAU']
            r = re.compile(r'[a-zA-Z]*')
            phone_dict = cmudict.dict()  # load cmudict from nltk
            for each_token in self.tokens:
                seq += phone_dict[each_token][0]  # add each letter's phone into the phone sequence
            for i in range(len(seq)):
                seq[i] = r.match(seq[i]).group(0)  # filter out digit number from phone sequence
            seq.append('PAU')
            return seq
        else:  # for normal speak
            seq = ['PAU']
            r = re.compile(r'([a-zA-Z]+|,|[.:?!])')
            r2 = re.compile(r'((\d+)\/(\d+)\/((19\d\d)|(\d\d)))|((\d+)\/(\d+))')
            phone_dict = cmudict.dict()
            for each_token in self.tokens:
                dt_all = r2.match(each_token)  # match the date components
                if each_token in [',','.',':','?','!']:  # if the token is punctuation
                    seq += ['PAU'] + [each_token] + ['PAU']  # add pause to sequence
                elif dt_all is not None:  # if the token is date
                    if each_token == dt_all.group(1):  # if date format is xx/xx/xxxx or xx/xx/xx
                        try:
                            if dt_all.group(5) is not None:  # if the year number is 19xx
                                dt = datetime.strptime(each_token[:-5], '%d/%m')
                            else:  # if the year number is xx
                                dt = datetime.strptime(each_token[:-3], '%d/%m')
                            seq += phone_dict[dt.strftime('%B').lower()][0]  # add month English word phone to seq
                            for each_word in date[int(dt_all.group(2))].split():
                                seq += phone_dict[each_word][0]  # add date English word phone to seq
                            words = num_to_word(int(dt_all.group(4)[-2:]))  # transfer year number to English word
                            seq += phone_dict['nineteen'][0]
                            for each_word in words:
                                seq += phone_dict[each_word][0]  # add year English word phone to seq
                        except ValueError:
                            print('the date {} is not correct'.format(each_token))
                    elif each_token == dt_all.group(7):  # if the date format is xx/xx
                        try:
                            dt = datetime.strptime(each_token, '%d/%m')
                            seq += phone_dict[dt.strftime('%B').lower()][0]  # add month English word phone to seq
                            for each_word in date[int(dt_all.group(8))].split():
                                seq += phone_dict[each_word][0]  # add date English word phone to seq
                        except ValueError:
                            print('the date {} is not correct'.format(each_token))
                else:  # if the token is normal English word
                    try:
                        seq += phone_dict[each_token][0]
                    except KeyError:
                        print("the word '{}' is not in dictionary and will be skipped".format(each_token))
            for i in range(len(seq)):
                seq[i] = r.match(seq[i]).group(0)  # filter out digit number from phone sequence
            seq.append('PAU')
            return seq

def smoother(data,time):  # use linspace array to smooth the beginning and end of audio data
    scale1 = np.linspace(0, 1, time * 16)
    scale2 = np.linspace(1, 0, time * 16)
    data[:time*16] = data[:time*16] * scale1
    data[-time*16:] = data[-time*16:] * scale2
    return data

if __name__ == "__main__":
    utt = Utterance(args.phrase[0])
    phone_seq = utt.get_phone_seq()
    diphone_seq = []
    for i in range(len(phone_seq)-1):
        if phone_seq[i] in [',','.',':','?','!']:
            diphone_seq.pop()  # delete diphone like 'xx-,' or 'xx-?'
            diphone_seq.append(phone_seq[i])  # add punctuation to seq for further punctuation operation
        else:
            diphone_seq.append(phone_seq[i]+'-'+phone_seq[i+1])
    diphone_synth = Synth(wav_folder=args.diphones)

    out = simpleaudio.Audio(rate=16000)
    temp_audio = simpleaudio.Audio(rate=16000)
    silence_200 = simpleaudio.Audio(rate=16000)
    silence_200.create_tone(0,3200,0)  # 200ms silence
    silence_400 = simpleaudio.Audio(rate=16000)
    silence_400.create_tone(0,6400,0)  # 400ms silence
    time = 10  # set cross-fade time to 10ms

    if args.crossfade is True:
        diphone_seq_size = len(diphone_seq)
        for i, diphone in enumerate(diphone_seq, 1):
            len = out.data.shape[0]
            temp_data = out.data[-time * 16:]
            if diphone is ',':  # add 200ms silence to output data
                out.data = np.append(out.data[:-time * 16], smoother(silence_200.data, time))
            elif diphone in ['.', ':', '?', '!']:  # add 400ms silence to output data
                out.data = np.append(out.data[:-time * 16], smoother(silence_400.data, time))
            elif i == 1:  # the first diphone file need not to smooth the beginning
                try:
                    temp_audio.load(diphone_synth.wavs[diphone])
                    scale2 = np.linspace(1, 0, time * 16)
                    temp_audio.data[-time * 16:] = temp_audio.data[-time * 16:] * scale2
                    out.data = np.append(out.data[:-time * 16], temp_audio.data)
                except KeyError:
                    print('diphone {} cannot be found and will be skipped'.format(diphone))
            elif i == diphone_seq_size:  # the last diphone file need not to smooth the end
                try:
                    temp_audio.load(diphone_synth.wavs[diphone])
                    scale1 = np.linspace(0, 1, time * 16)
                    temp_audio.data[:time * 16] = temp_audio.data[:time * 16] * scale1
                    out.data = np.append(out.data[:-time * 16], temp_audio.data)
                except KeyError:
                    print('diphone {} cannot be found and will be skipped'.format(diphone))
            else:
                try:
                    temp_audio.load(diphone_synth.wavs[diphone])
                    out.data = np.append(out.data[:-time * 16], smoother(temp_audio.data, time))
                    out.data[len - time * 16:len] += temp_data
                except KeyError:
                    print('diphone {} cannot be found and will be skipped'.format(diphone))
    else:  # no cross-fade
        for diphone in diphone_seq:
            if diphone is ',':
                out.data = np.append(out.data, silence_200.data)
            elif diphone in ['.', ':', '?', '!']:
                out.data = np.append(out.data, silence_400.data)
            else:
                try:
                    temp_audio.load(diphone_synth.wavs[diphone])
                    out.data = np.append(out.data, temp_audio.data)
                except KeyError:
                    print('diphone {} cannot be found and will be skipped'.format(diphone))

    if args.volume is not None:  # adjust the volume
        out.rescale(args.volume/100)
    if args.play is True:  # play audio
        out.play()
    if args.outfile is not None:  # save the file
        out.save("./{}".format(args.outfile))


