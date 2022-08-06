import argparse
import numpy as np

def parse_args() -> tuple:
    """
    Collects all arguments passed in via command line and returns the appropriate 
    data. Returns:

        (1) train_data : A list [X1, X2, ..., XN], where each element Xi is a training 
            example represented as a list of tuples:

                Xi = [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]

            For example:

                train_data = [[(None, "<START>"), ("fish", "D"), (next_tuple)], [next_train_example], ...]

            Note that this function automatically includes the "<START>" and 
            "<END>" tags for you.

        (2) words_dict : A dictionary with keys of type str and values of type int. 
            Keys are words and values are their indices. For example:

                words_dict["hi"] == 99

        (3) tags_dict : A dictionary with keys of type str and values of type int.
            Keys are tags and values are their indices. For example:

                tags_dict["<START>"] == 0

        (4) emit : A string representing the path of the output hmmemit.txt file.
        
        (5) trans : A string representing the path of the output hmmtrans.txt file.
    
    Usage:
        train_data, word_dict, tags_dict, emit, trans = parse_args()
    """
    # Define a parser
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input', type=str,
                        help='path to training input .txt file')
    parser.add_argument('index_to_word', type=str,
                        help='path to index_to_word.txt file')
    parser.add_argument('index_to_tag', type=str,
                        help='path to index_to_tag.txt file')
    parser.add_argument('emit', type=str,
                        help='path to store the hmmemit.txt file')
    parser.add_argument('trans', type=str,
                        help='path to store the hmmtrans.txt file')
    
    args = parser.parse_args()
    

    # Create train data
    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for i in range(len(examples)):
            example = examples[i].split("\n")
            train_data.append([t.split("\t") for t in example])
    train_data = [[(None, "<START>")] + elem + [(None, "<END>")] for elem in train_data]

    # making dictionary of words to index
    words_dict = {}
    with open(args.index_to_word, "r") as words_indices:
        i = 0
        for line in words_indices:
            words_dict[line.rstrip()] = i
            i += 1

    # making dictionary of words to tags
    tags_dict = {}
    with open(args.index_to_tag, "r") as tags_indices:
        j = 0
        for line in tags_indices:
            tags_dict[line.rstrip()] = j
            j += 1
    
    return train_data, words_dict, tags_dict, args.emit, args.trans




def indexify(train_data):
    # modify the trainning to index
    train_index = []
    trainsize = len(train_data)
    for i in range (trainsize):
        # get sentence
        sentence = train_data[i]
        sentence_i = []
        for j in range (len(sentence)):
            # get word
            word = sentence[j]
            word_i = []
            # the start and end word are none
            if word[0]!= None:
                word_i.append(words_dict[word[0]])
            else:
                word_i.append(None)
            word_i.append(tags_dict[word[1]])
            sentence_i.append(word_i)

        train_index.append(sentence_i)
    return train_index

def increaseA(train_index,emitA):
    for sentence in train_index:
        for word in sentence:
            if word[0]!=None:
                # state as the row
                row = word[1]
                # word is the col
                col = word[0]
                emitA[row][col]+=1
    return emitA

def increaseB(train_index,transB):
    for sentence in train_index:
        # for each sentence, get the second col (states)
        s = np.array(sentence)
        states = s[:, 1]
        states_size = len(states)
        # iterate all the transitions
        for i in range (0, states_size-1):
            changerow = states[i]
            changecol = states[i+1]
            transB[changerow][changecol]+=1
    return transB


if __name__ == "__main__":
    
    train_data, words_dict, tags_dict, emit, trans = parse_args()
    train_data = train_data[:10000]


    
    

    # Initialize emit (A) and trans (B) matrices
    stateN = len(tags_dict)
    wordN = len(words_dict)
    # A is state by word
    emitA = np.zeros((stateN,wordN))
    # B is state by state
    transB = np.zeros((stateN,stateN))
    

    
    # Iterate through the data and increment the appropriate cells in the matrices
    
    # get the indexed train
    train_index = indexify(train_data)
    # increment A
    incre_A = increaseA(train_index,emitA)

    incre_B = increaseB(train_index,transB)





    # Add a pseudocount of 1
    # add one to all elements for A except the first and the last row
    incre_A[:,:]+=1

    # add one to all elements for B excpet the last row and first col
    incre_B[:-1,1:] += 1


    # Convert the rows of A and B to probability distributions. Each row 
    # of A and B should sum to 1 (except for the rows mentioned below). 
    rawA = incre_A/incre_A.sum(axis=1,keepdims=True)
    # manual adjust
    rawA[0,:] = 1
    rawA[-1,:] = 1
    rawB = incre_B/incre_B.sum(axis=1,keepdims=True)
    normalA = np.nan_to_num(rawA)
    normalB = np.nan_to_num(rawB)

    # Please note that:
    # 
    #   B[:, tags_dict["<START>"]] == 0 since nothing can transition to <START>
    #   B[tags_dict["<END>"], :]   == 0 since <END> can't transition to anything
    #   A[tags_dict["<START>"], :] == 1 since <START> emits nothing; setting to 1 makes forwardbackward easier (as opposed to setting to 0)
    #   A[tags_dict["<END>"], :]   == 1 since <END> emits nothing
    #
    # You should manually ensure that the four conditions above hold (e.g. by 
    # manually setting the rows/columns to the desired values and ensuring that 
    # the other rows not mentioned above remain probability distributions)

    # Save the emit and trans matrices (the reference solution uses np.savetxt 
    # with fmt="%.18e")

    np.savetxt(emit, normalA,fmt="%.18e")
    np.savetxt(trans, normalB,fmt="%.18e")