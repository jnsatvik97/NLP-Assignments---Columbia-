from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        #print(words)
        #print(pos)

        while state.buffer: 

            # TODO: Write the body of this loop for part 4

            # print("The Buffer is", state.buffer)
            # print("The Stack is", state.stack)

            inp_vector = self.extractor.get_input_representation(words, pos, state)

            # final_vector = np.zeros(1)
            # final_vector[0] = inp_vector

            inp_vector_trans = inp_vector.transpose()

            inp_vector_trans = inp_vector_trans.reshape(1, -1)

            #print(inp_vector_trans.shape)

            preds = self.model.predict(inp_vector_trans)

            # preds[::-1].sort()

            # print(preds)

            predsList = list(preds[0])

            #print(predsList)

            predsList = np.argsort(predsList)

            # print(predsList)

            i = len(predsList)-1

            # print(i)

            # print(self.output_labels)

            while i >= 0:

                action = self.output_labels[predsList[i]]

                # print(action)

                if len(state.stack) == 0 and (action[0] == "left_arc" or action[0] == "right_arc"):
                    i = i - 1

                elif len(state.buffer) == 1 and len(state.stack) != 0 and action[0] == "shift":
                    i = i - 1

                elif len(state.stack) != 0 and state.stack[-1] == 0 and action[0] == "left_arc":
                    i = i - 1

                else:
                    # print("Reaches Else")

                    if action[0] == "left_arc":
                        state.left_arc(action[1])

                    elif action[0] == "right_arc":
                        state.right_arc(action[1])

                    else:
                        state.shift()

                    break
            # print("Reaches end of inner")

        # print("Reaches Here")

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result

        # create feature representation of input
        # Run model on feature representation to get 91 dim output vector
        # Sort the vector in desc order
        # Iterate from beginning tii you find valid transition for the given state
        # Add transition (triple) into state.deps
        # Change the state according to the transition
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
