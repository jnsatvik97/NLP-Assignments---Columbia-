#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import string
import numpy as np

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []

    LemmaList = wn.lemmas(lemma, pos=pos)

    print(LemmaList)

    for lemmaTemp in LemmaList:
        LemmaListNew = lemmaTemp.synset().lemmas()

        print(LemmaListNew)

        for l in LemmaListNew:
            name = l.name()
            print(name)
            if name != lemma:
                if name not in possible_synonyms:
                    if "_" in name:
                        name = name.replace("_", " ")
                    possible_synonyms.append(name)

    return possible_synonyms



def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):

    countFreq = {}

    lemma = context.lemma
    pos = context.pos

    LemmaList = wn.lemmas(lemma, pos=pos)

    for lemmaTemp in LemmaList:

        # count = lemmaTemp.count()

        LemmaListNew = lemmaTemp.synset().lemmas()

        for l in LemmaListNew:
            name = l.name()
            name = name.lower()
            if name != lemma:
                if "_" in name:
                    name = name.replace("_", " ")
                if name not in countFreq:
                    countFreq[name] = l.count()
                else:
                    countFreq[name] += l.count()

    maxVal = -1
    maxWord = None

    for k, v in countFreq.items():

        if v > maxVal:
            maxVal = v
            maxWord = k

    return maxWord

    #return None # replace for part 2

def wn_simple_lesk_predictor(context):

    lemma = context.lemma
    pos = context.pos

    LemmaList = wn.lemmas(lemma, pos=pos)

    countOverlaps = {}

    for lemmaTemp in LemmaList:

        s = lemmaTemp.synset()

        listOfWords = []

        definitionList = tokenize(s.definition())

        for d in definitionList:
            listOfWords.append(d)

        for example in s.examples():
            examplesList = tokenize(example)

            for e in examplesList:
                listOfWords.append(e)

        for syn in s.hypernyms():

            defHyperList = tokenize(syn.definition())

            for d in defHyperList:
                listOfWords.append(d)

            for example in syn.examples():
                examplesList = tokenize(example)

                for e in examplesList:
                    listOfWords.append(e)


        newListWords = list(set(listOfWords))

        for word in stopwords.words('english'):
            if word in newListWords:
                newListWords.remove(word)

        contextWords = context.left_context + context.right_context

        newContextWords  = list(set(contextWords))

        overlap = len(list(set(newListWords).intersection(newContextWords)))

        countOverlaps[s] = overlap

    Synset = None

    OverLapList = sorted(countOverlaps.items(), key=lambda x: x[1], reverse=True)

    if OverLapList[0][1] == 0 or OverLapList[0][1] == OverLapList[1][1]:
        maxcountSyn = -1
        lemmaTemp = None
        for l in LemmaList:
            count = l.count()
            if count > maxcountSyn:
                maxcountSyn = count
                lemmaTemp = l
        Synset = lemmaTemp.synset()

    else:
        Synset = OverLapList[0][0]

    maxlexFreq = -1
    name = None
    for lexeme in Synset.lemmas():
        n = lexeme.name()
        if n != lemma:
            if lexeme.count() > maxlexFreq:
                maxlexFreq = lexeme.count()
                name = n

    return name


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def cos(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def predict_nearest(self, context):

        lemma = context.lemma
        pos = context.pos

        similarity = {}

        LemmaList = wn.lemmas(lemma, pos=pos)

        for lemmaTemp in LemmaList:

            LemmaListNew = lemmaTemp.synset().lemmas()

            # print(LemmaListNew)

            for l in LemmaListNew:
                name = l.name()
                # print(name)
                if name != lemma:
                    if name not in self.model.vocab:
                        continue
                    #if "_" in name:
                       # name = name.replace("_", " ")
                        # continue
                    if name not in similarity:
                        similarity[name] = self.model.similarity(name, lemma)

        SimilarityList = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

        return SimilarityList[0][0]

        # return None # replace for part 4

    def predict_nearest_with_context(self, context):

        lemma = context.lemma
        pos = context.pos

        left_context = context.left_context
        right_context = context.right_context

        stop_words = stopwords.words('english')

        # print(stop_words)

        for word in stop_words:
            if word in left_context:
                left_context.remove(word)
            if word in right_context:
                right_context.remove(word)

        if len(left_context) > 5:
            left_context = left_context[-5:]

        if len(right_context) > 5:
            right_context = right_context[:5]

        resArray = np.zeros(300)

        for word in left_context:
            if word in self.model.vocab:
                resArray = np.add(resArray, self.model.wv[word])

        for word in right_context:
            if word in self.model.vocab:
                resArray = np.add(resArray, self.model.wv[word])

        if lemma in self.model.vocab:
            resArray = np.add(resArray, self.model.wv[lemma])

        similarity = {}

        LemmaList = wn.lemmas(lemma, pos=pos)

        for lemmaTemp in LemmaList:

            LemmaListNew = lemmaTemp.synset().lemmas()

            # print(LemmaListNew)

            for l in LemmaListNew:
                name = l.name()
                # print(name)
                if name != lemma:
                    if name not in self.model.vocab:
                        continue
                    # if "_" in name:
                    # name = name.replace("_", " ")
                    # continue
                    if name not in similarity:
                        nameVec = self.model.wv[name]
                        similarity[name] = self.cos(resArray, nameVec)

        SimilarityList = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

        return SimilarityList[0][0]


    #  This is the solution for Part 6: Reduced the length of the context to +-2 words around the target word (instead of +-5) which
    # resulted in the best Precision and Recall obtained among all Parts. The logic behind this improvement would be that now we are looking at
    # more nearer (and hence more relevant) context words which would lead to a better similarity with the synonyms

    def best_predictor(self, context):

        lemma = context.lemma
        pos = context.pos

        left_context = context.left_context
        right_context = context.right_context

        stop_words = stopwords.words('english')

        # print(stop_words)

        for word in stop_words:
            if word in left_context:
                left_context.remove(word)
            if word in right_context:
                right_context.remove(word)

        if len(left_context) > 2:
            left_context = left_context[-2:]

        if len(right_context) > 2:
            right_context = right_context[:2]

        resArray = np.zeros(300)

        for word in left_context:
            if word in self.model.vocab:
                resArray = np.add(resArray, self.model.wv[word])

        for word in right_context:
            if word in self.model.vocab:
                resArray = np.add(resArray, self.model.wv[word])

        if lemma in self.model.vocab:
            resArray = np.add(resArray, self.model.wv[lemma])

        similarity = {}

        LemmaList = wn.lemmas(lemma, pos=pos)

        for lemmaTemp in LemmaList:

            LemmaListNew = lemmaTemp.synset().lemmas()

            # print(LemmaListNew)

            for l in LemmaListNew:
                name = l.name()
                # print(name)
                if name != lemma:
                    if name not in self.model.vocab:
                        continue
                    # if "_" in name:
                    # name = name.replace("_", " ")
                    # continue
                    if name not in similarity:
                        nameVec = self.model.wv[name]
                        similarity[name] = self.cos(resArray, nameVec)

        SimilarityList = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

        return SimilarityList[0][0]


    #  Also tried the below approach for Part 6 but it did not generate good results: Took the average of the embeddings of the
    # context word and then computed element wise product with the target word to generate a 300 dimension vector which was then
    # checked for similarity with the potential synonyms

    def predict_nearest_with_contextAverage(self, context):

        lemma = context.lemma
        pos = context.pos

        left_context = context.left_context
        right_context = context.right_context

        stop_words = stopwords.words('english')

        for word in stop_words:
            if word in left_context:
                left_context.remove(word)
            if word in right_context:
                right_context.remove(word)

        # if len(left_context) > 5:
            # left_context = left_context[-5:]

        # if len(right_context) > 5:
            # right_context = right_context[:5]

        resArray = np.zeros(300)

        countLeft = 0
        for word in left_context:
            if word in self.model.vocab:
                resArray += self.model.wv[word]
                countLeft += 1

        countRight = 0
        for word in right_context:
            if word in self.model.vocab:
                resArray += self.model.wv[word]
                countRight += 1



        if lemma in self.model.vocab:
            resArray = np.multiply(self.model.wv[lemma], resArray)

        similarity = {}

        LemmaList = wn.lemmas(lemma, pos=pos)

        for lemmaTemp in LemmaList:

            LemmaListNew = lemmaTemp.synset().lemmas()

            # print(LemmaListNew)

            for l in LemmaListNew:
                name = l.name()
                # print(name)
                if name != lemma:
                    if name not in self.model.vocab:
                        continue
                    # if "_" in name:
                    # name = name.replace("_", " ")
                    # continue
                    if name not in similarity:
                        nameVec = self.model.wv[name]
                        similarity[name] = self.cos(resArray, nameVec)

        SimilarityList = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

        return SimilarityList[0][0]



if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # ResList = get_candidates('slow', 'a')
    # print(ResList)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.best_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))


