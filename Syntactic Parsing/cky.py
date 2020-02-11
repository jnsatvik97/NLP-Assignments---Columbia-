"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
import numpy as np

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        ResDict = defaultdict(list)

        n = len(tokens)

        for i in range(0, n):
            t = (i, i + 1)
            ResDict[t]
            #print(tokens[i])
            tempTuple=(tokens[i],)
            #print(tempTuple)
            if tempTuple in self.grammar.rhs_to_rules:
                #print("Reaches here")
                #t = (i, i+1)
                #t.append(i)
                #t.append(i+1)
                #ResDict[t] = ()
                for l in self.grammar.rhs_to_rules[tempTuple]:
                    ResDict[t].append(l[0])

        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i+length
                t = (i, j)
                ResDict[t]
                for k in range(i+1, j):
                    for b in ResDict[(i, k)]:
                        for c in ResDict[(k, j)]:
                            tNew = (b, c)
                            if tNew in self.grammar.rhs_to_rules:
                                for l in self.grammar.rhs_to_rules[tNew]:
                                    if l[0] not in ResDict[t]:
                                        ResDict[t].append(l[0])

        #print(ResDict)

        for s in ResDict[(0, n)]:
            if s == self.grammar.startsymbol:
                return True

        return False

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        # table= None
        # probs = None

        BackDict = defaultdict(dict)
        ProbDict = defaultdict(dict)

        n = len(tokens)

        for i in range(0, n):
            t = (i, i + 1)
            BackDict[t]
            ProbDict[t]

            tempTuple = (tokens[i],)
            #print(tempTuple)
            if tempTuple in self.grammar.rhs_to_rules:
                for l in self.grammar.rhs_to_rules[tempTuple]:
                    ProbDict[t][l[0]] = np.log2(l[2])     # probabilities in log2 format
                    BackDict[t][l[0]] = tokens[i]

        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                t = (i, j)
                BackDict[t]
                ProbDict[t]
                for k in range(i + 1, j):
                    for b in BackDict[(i, k)].keys():
                        for c in BackDict[(k, j)].keys():
                            tNew = (b, c)
                            if tNew in self.grammar.rhs_to_rules:
                                for l in self.grammar.rhs_to_rules[tNew]:
                                    if l[0] not in BackDict[t]:
                                        BackDict[t][l[0]] = ((b,i,k), (c,k,j))
                                        ProbDict[t][l[0]] = np.log2(l[2]) + ProbDict[(i, k)][b] + ProbDict[(k, j)][c]

                                    else:
                                        if np.log2(l[2]) + ProbDict[(i, k)][b] + ProbDict[(k, j)][c] > ProbDict[t][l[0]]:
                                            ProbDict[t][l[0]] = np.log2(l[2]) + ProbDict[(i, k)][b] + ProbDict[(k, j)][c]
                                            BackDict[t][l[0]] = ((b, i, k), (c, k, j))

        return BackDict, ProbDict

def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    # Return a Parse Tree

    # Chart here is BackDict

    if j-i == 1:
        t = (nt, chart[i, j][nt])
        return t

    t1 = get_tree(chart, chart[i, j][nt][0][1], chart[i, j][nt][0][2], chart[i, j][nt][0][0])

    t2 = get_tree(chart, chart[i, j][nt][1][1], chart[i, j][nt][1][2], chart[i, j][nt][1][0])

    Finalt = (nt, t1, t2)

    return Finalt


if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file)

        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']

        print("The list of tokens is in language: ", parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)

        print("This is the backpointer table: ", table)
        print("This is the probability table: ", probs)

        assert check_table_format(table)
        assert check_probs_format(probs)

        print("This is the parse tree: ", get_tree(table, 0, len(toks), grammar.startsymbol))

