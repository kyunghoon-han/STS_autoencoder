import random, pickle
# Hanguls

onsets = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ",
    "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ")
consonants = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ",
    "ㅃ", "ㅅ", "ㅆ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ")
nuclei = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
    "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")
codas = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
    "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
    "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ")

extended_hanguls = set(onsets+nuclei+codas)

def is_Hangul(a_char):
    """Confirms whether the input is a Hangeul or not"""
    return 0xAC00 <= ord(a_char[:1]) <= 0xD7A3

def Hangul_decomposition(a_hangul):
    """ Decomposes the input Hangul character into onset, nucleus and coda.
    If the character lacks a coda, its supposed coda is represented by a Z
    """
    if not is_Hangul(a_hangul):
        return a_hangul
    num_val = ord(a_hangul) - 0xAC00
    onset = num_val // (21*28)
    nucleus = num_val % (21*28) // 28
    coda = num_val % 28
    if coda == 0:
        return onsets[onset] + nuclei[nucleus]
    else:
        return onsets[onset] + nuclei[nucleus] + codas[coda]

def sentence_to_decomposition(a_sentence):
    output = ""
    for i in a_sentence:
        output = output + Hangul_decomposition(i)
    return output

def encoder_generator(vec=False):
    extra_letters = set(" '!().,\"*?'")
    letters = list(extra_letters.union(extended_hanguls))
    random.shuffle(letters)
    len_vectors = len(letters)
    hangul_dict = {}
    if vec:
        for i in range(len_vectors):
            tmp_zeros = [0]*len_vectors
            tmp_zeros[i] = 1
            hangul_dict[letters[i]] = tmp_zeros
    else:
        for i in range(len_vectors):
            hangul_dict[letters[i]] = i
    
    a_file = open("encoded_hangul.pkl","wb")
    pickle.dump(hangul_dict,a_file)
    a_file.close()

    return hangul_dict

def sentence2vec(a_sentence, decom_bool=True,hangul_dict = {},filename=".encoded_hangul.pkl", 
                from_file=False, vec=False):
    a_sentence = a_sentence.replace("“","\"").replace("”","\"").replace("/"," ")
    a_sentence = a_sentence.replace("’","'").replace("‘","'").replace("`","'")
    a_sentence = a_sentence.replace("@","*").replace("#","*").replace(":","*")
    a_sentence = a_sentence.replace("+","*").replace("-","*").replace("=","*")
    a_sentence = a_sentence.replace("~","*").replace("&","*").replace("%","*")
    a_sentence = a_sentence.replace("[","(").replace("{","(").replace("#","*")
    a_sentence = a_sentence.replace("]",")").replace("}",")").replace("_","*")
    # bring up the dictionary
    if len(hangul_dict) != 0:
        pass # when a desired dictionary is input
    elif from_file:
        with open(filename,'rb') as f:
            hangul_dict = pkl.load(f)
    else:
        if not vec:
            hangul_dict = encoder_generator()
        else:
            hangul_dict = encoder_generator(vec=True)
    # now a for loop
    output = []
    if decom_bool:
        sentence_decom = sentence_to_decomposition(a_sentence)
    for a in sentence_decom:
        #print("this is a: ", a)
        #print("what would be hangul_dict[a]? ",hangul_dict[a])
        output.append(hangul_dict[a])
    return output, hangul_dict

if __name__=='__main__':
    encoder_generator()
