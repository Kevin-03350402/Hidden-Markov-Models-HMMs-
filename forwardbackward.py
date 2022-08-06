import argparse
import numpy as np
import sys


def parse_args() -> tuple:
    """
    Collects all arguments from command line and returns data. Returns:

        (1) validation_data : A list [X1, X2, ..., XN], where each Xi is a validation 
            example:

                Xi = [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]
            
            This function automatically includes <START> and <END> tags for you.
            
        (2) words_dict : A dictionary mapping words (str) to indices (int).

        (3) tags_dict : A dictionary mapping tags (str) to indices (int).

        (4) emit : A numpy matrix containing the emission probabilities.

        (5) trans : A numpy matrix containing the transition probabilities.

        (6) prediction_file : A string indicating the path to write predictions to.

        (7) metric_file : A string indicating the path to write metrics to.
    
    Usage:
        validation_data, words_dict, tags_dict, emit, trans, prediction_file, metric_file = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_input', type=str,
                        help='path to validation input .txt file')
    parser.add_argument('index_to_word', type=str,
                        help='path to index_to_word.txt file')
    parser.add_argument('index_to_tag', type=str,
                        help='path to index_to_tag.txt file')
    parser.add_argument('emit', type=str,
                        help='path to the learned hmmemit.txt file')
    parser.add_argument('trans', type=str,
                        help='path to the learned hmmtrans.txt file')
    parser.add_argument('prediction_file', type=str,
                        help='path to store predictions')
    parser.add_argument('metric_file', type=str,
                        help='path to store metrics')
    
    args = parser.parse_args()

    # Create train data
    validation_data = list()
    with open(args.validation_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for i in range(len(examples)):
            example = examples[i].split("\n")
            validation_data.append([t.split("\t") for t in example])
    validation_data = [[(None, "<START>")] + elem + [(None, "<END>")] for elem in validation_data]

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
    
    emit = np.loadtxt(args.emit, delimiter=" ")

    trans = np.loadtxt(args.trans, delimiter=" ")

    return validation_data, words_dict, tags_dict, emit, trans, args.prediction_file, args.metric_file

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

def logsumexp(x):
    """
    Computes log (sum over all i (e^{x_i})) by using the log-sum-exp trick. You 
    may find it helpful to define a logsumexp function for a matrix X as well. 
    Please note that, when all elements of the vector x are -np.inf, your 
    logsumexp function should return -np.inf and not np.nan.

    Arguments:

        x : A numpy array of dimension 1 (e.g. a vector, or a list)
    """
    # all elements are -inf
    m = max(x)
    if m == -np.inf:
        m=0
    removed_m = x-m
    # exp of the array with max removed
    exp_r_m = np.exp(removed_m)
    # sum them, and log the sum, add m
    return m + np.log(np.sum(exp_r_m))
    
    
def log_inf(x):
    return np.log(x) if x>0 else -float('Inf')   

def forwardbackward(x, logtrans, logemit, cap_t, cap_j,words_dict, tags_dict):
    """
    Your implementation of the forward-backward algorithm. Remember to compute all 
    values in log-space and use the log-sum-exp trick!

    Arguments:

        x : A list of words

        logtrans : The log of the transition matrix

        logemit : The log of the emission matrix

        words_dict : A dictionary mapping words to indices

        tags_dict : A dictionary mapping tags to indices

    Returns:

        Your choice! The reference solution returns a list containing the predicted 
        tags for each word in x and the log-probability of x.
    
    """
    # initialize alpha0
    alpha = np.zeros((cap_j,cap_t))
    alpha[0][0] = 1
    alpha[:,0] = np.log(alpha[:,0])
    # iterate through
    for t in range(1, cap_t-1):
        
        wordt = x[t]

        for j in range (0, cap_j):
           
            alphat1k = alpha[:,t-1]

            bj = logtrans[:,j]
            
            v = alphat1k+bj
            log_alphajt = logemit[j][wordt]+logsumexp(v)
            alpha[j][t] = log_alphajt


    beta = np.zeros((cap_j,cap_t))
    beta[-1][-1] = 1
    beta[:,-1] = np.log(beta[:,-1])
    # iterate through
    beta[:,cap_t-2] = logtrans[:,-1]
    
    for t in range(cap_t-3,0,-1):
        
        wordt = x[t+1]
         
        for j in range (cap_j-1,0,-1):
            fir = logemit[:,wordt]
            sec = beta[:,t+1]
            third = logtrans[j,:]
            v = fir+sec+third
            log_betajt = logsumexp(v)
            

          
            beta[j][t] = log_betajt
    print(np.exp(alpha))
    print(np.exp(beta))
    wordp = []  
    
    states = list(tags_dict.keys())

    for t in range (1,cap_t-1):
        pyt = alpha[:,t]+beta[:,t]
        index = np.argmax(pyt)
        wordp.append(states[index])
    wordp.append(logsumexp(alpha[:,cap_t-2]+beta[:,cap_t-2]))

    return wordp

if __name__ == '__main__':

    validation_data, words_dict, tags_dict, emit, trans, predict_file, metric_file = parse_args()

    # convert all validation into index
    words = list(words_dict.keys())
    validation_data = indexify(validation_data)
    # take log on both emit and trans
    logemit = np.log(emit)
    logtrans = np.log(trans)


    
    
   

    # Iterate over the sentences; for each list of words x, compute its most likely 
    # tags and its log-probability using your forwardbackward function
    totalword = 0
    totalr = 0
    totalsentence = 0
    totallog = 0
    wordpre = open(predict_file,"w+")
    metric = open(metric_file,"w+")

    for sentence in validation_data: # sentence looks like [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]
        x = np.array([word for word, tag in sentence]) # x looks like [word1, word2, ..., wordM]
        y = np.array([tag for word, tag in sentence]) # y looks like [tag1, tag2, ..., tagM]
        xw = x[1:-1]
        yw = y[1:-1]
        totalword += len(xw)
        totalsentence+=1
        # T is just the length of the length of the sentence 

        # because total word length is len(x), the index of the last word is length-2
        cap_t = len(x)
        
        # J = number of states, the last state should have index j-1
        cap_j = len(tags_dict)
        res = forwardbackward(x, logtrans, logemit, cap_t, cap_j,words_dict,tags_dict)
        resp = res[:-1]
        totallog += res[-1]
        for k in range (len(xw)):
            wordpre.write(f'{words[xw[k]]}\t')

            wordpre.write(f'{resp[k]}\n')
            
        	
        wordpre.write('\n')
        
        for j in range (len(xw)):
            if yw[j]== tags_dict[resp[j]]:
                totalr+=1
    
    avglog = (totallog/totalsentence)
    accu = (totalr/totalword)
    metric.write(f'Average Log-Likelihood: {avglog:.15f}\n')
    metric.write(f'Accuracy: {accu:.16f}\n')
    wordpre.close()
    metric.close()


    
    

    

        

    # Compute the average log-likelihood of all x and the accuracy of your 
    # HMM. When computing the accuracy, you should *NOT* include the first and 
    # last tags, since these are always <START> and <END>. If you're using the 
    # code above, this means that you should only consider y[1:-1] when computing 
    # the accuracy. The accuracy is computed as the total number of correct tags 
    # across all validation sentences divided by the total number of tags across 
    # all validation sentences.

    # Write the predictions (as words and tags, not indices) and the metrics. 
    # The reference solution doesn't use any special formatting when writing 
    # the metrics.