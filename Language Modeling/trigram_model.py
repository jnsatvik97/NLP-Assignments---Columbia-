import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np


"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

# Returns the different types of words in the document

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


# returns the total number of words in the document


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    sequence.append("STOP")
    if n == 1:
        new_list = ["START"]
    else:
        new_list=["START"]*(n-1)
    new_list.extend(sequence)

    # THIS IS INEFFICIENT CODE
    ###############################
    #print(new_list)

    #resultList=[]
    # for i in range(O:len(sequence))
    #i=0
    #while((i+n-1)!=len(new_list)):
    #    resultList.append(tuple(new_list[i:i+n]))
     #   i=i+1
    ###############################

    resultList = list(zip(*(new_list[i:] for i in range(n))))
    return resultList


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)

        self.lexicon = get_lexicon(generator)

        self.lexicon.add("UNK")

        self.lexicon.add("START")

        self.lexicon.add("STOP")

        #self.num_words = get_num_words(generator)
    
        # Now iterate through the corpus again - this time lexicon is there so 'UKN' are added
        generator = corpus_reader(corpusfile, self.lexicon)

        self.get_num_words(generator)

        generator = corpus_reader(corpusfile, self.lexicon)

        self.count_ngrams(generator)



    def get_num_words(self, corpus):
        word_count = 0
        for sentence in corpus:
            #print(sentence)
            word_count=word_count+len(sentence)
                # print(word_counts)
        self.word_count = word_count


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        #for i in range(0,3):
        for sentence in corpus:
            for i in range(0, 3):
                newList = get_ngrams(sentence, i+1)
                #print(list)
                for t in newList:
                    if i == 0:
                        self.unigramcounts[t] += 1
                    elif i == 1:
                        self.bigramcounts[t] += 1
                    else:
                        self.trigramcounts[t] += 1

        #for i in range(0,3):
            #tempList = list(map(get_ngrams(i), corpus))

         #   tempList=[get_ngrams(sentence, i+1) for sentence in corpus]



        #print(self.bigramcounts)
        #return


        return

        ##Your code here

        #return



    # This is calculated from the training data

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if(self.bigramcounts[trigram[0:2]]!=0):
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[0:2]]

        return 0.0

    # This is calculated from the training data

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        # Make a tuple which consists of the unigram word + ','
        t = (bigram[0],)
        if (self.unigramcounts[t]!= 0):
            return self.bigramcounts[bigram] / self.unigramcounts[t]

        return 0.0

    # This is calculated from training data
    # Probability of finding that unigram calculated from training data
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        # This has to be the training file always
        #dev_corpus = corpus_reader('brown_train.txt', self.lexicon)


        # every time it wants to calculate unigram prob it has to recompute total number of words - Resolved this
        numberWords= self.word_count

        #print(numberWords)

        return self.unigramcounts[unigram]/numberWords   # Assuming that in this estimation you divide by number of total words

        #return 0.0


    #  This is an optional task

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        t = (trigram[2],)
        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:3]) + lambda3*self.raw_unigram_probability(t)

    # This is calculated for testing data on the basis of training data
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        print(sentence)
        resList = get_ngrams(sentence, 3)

        # This is inefficient code segment

        #ProbSentence = 0

        #for t in resList:

         #   if self.smoothed_trigram_probability(t) != 0:
         #       prob = math.log2(self.smoothed_trigram_probability(t))
         #   else:
         #       prob = 0

         #  ProbSentence = ProbSentence+prob



        tempList = list(map(self.smoothed_trigram_probability, resList))
        tempList = np.log2(tempList)
        ProbSentence = np.sum(tempList)

        return ProbSentence



    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        sumProbSentences=0

        M=0     # represents total words in testing corpus

        for sentence in corpus:

            # code reaches here

            #print(sentence)
            M = M+len(sentence)
            sumProbSentences = sumProbSentences + self.sentence_logprob(sentence)
            #print(sumProbSentences)

        l = sumProbSentences/M

        #print(l)

        return pow(2.0, -1.0*l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if(pp1<pp2):
                correct=correct+1
            total=total+1
    
        for f in os.listdir(testdir2):
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))

            if (pp1 < pp2):
                correct = correct + 1
            total = total + 1
        
        return (correct/total)*100

if __name__ == "__main__":

    model = TrigramModel('brown_train.txt')


    print(model.word_count)


    sequence=["natural","language","processing"]
    n= 5
    Testinglist=get_ngrams(sequence,n)
    print(Testinglist)


    print(model.unigramcounts[('the',)])   # This is working fine

    print(model.bigramcounts[('START','the')])  # This is working fine

    print(model.trigramcounts[('START', 'START', 'the')])  # This is also working fine


    print(model.raw_bigram_probability(('START','the')))    # This is working fine

    print(model.raw_trigram_probability(('START','START','the')))  # This is also working fine

    print(model.raw_unigram_probability(('the',)))   # This is also working fine



    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:

    # reading the test file using the lexicon from the training data

    #dev_corpus = corpus_reader('brown_test.txt', model.lexicon)
    #dev_corpus = corpus_reader('brown_train.txt', model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', "train_low.txt", "test_high", "test_low")
    print(acc)

