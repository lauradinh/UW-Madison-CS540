import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding = 'utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    bow[None] = 0
    with open (filepath, 'r', encoding ='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if not word in bow and len(word) > 0:
                if word in vocab:
                    bow[word] = 1
                else:
                    bow[None] += 1
            elif len(word) > 0:
                bow[word] += 1
    if bow[None] == 0:
        bow.pop(None)
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    num_of_2016 = 0;
    num_of_2020 = 0;
    
    #counting for docs in 2016
    for doc in training_data:
            if doc['label'] == '2016':
                num_of_2016 += 1;
                
    #counting for docs in 2016
    for doc in training_data:
            if doc['label'] == '2020':
                num_of_2020 += 1;
    
    total_files = num_of_2016 + num_of_2020
    
    prob_of_2016 = (num_of_2016 + smooth) / (total_files + 2)
    prob_of_2020 = (num_of_2020 + smooth) / (total_files + 2)
    
    logprob['2020'] = math.log(prob_of_2020)
    logprob['2016'] = math.log(prob_of_2016)
    
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    vocab_length = len(vocab)
    word_count_of_label = 0
    

    #counts total word count
    for doc in training_data:
        if doc['label'] == label:
            for word in doc['bow']:
                word_count_of_label += doc['bow'][word]
    
    #intializaes word counts
    word_count = {}
    for word in vocab: 
        word_count[word] = 0
    word_count[None] = 0
    
    # finds the counts of every word
    for doc in training_data:
        for word in doc['bow']:
            if word in vocab and doc['label'] == label:
                word_count[word] += doc['bow'][word]
            #OOV
            elif word not in vocab and doc['label'] == label:
                word_count[None] += doc['bow'][word]
    
    for word in word_count:
        word_prob[word] = math.log((word_count[word] + smooth) / (word_count_of_label + vocab_length + smooth))

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(retval['vocabulary'], training_directory)
    
    retval['log prior'] = prior(training_data, label_list)
    
    retval['log p(w|y=2016)'] = p_word_given_label(retval['vocabulary'], training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(retval['vocabulary'], training_data, '2020')
    return retval
    
    return training_data

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    bow = create_bow(model['vocabulary'], filepath)
    
    sum_of_p_given_2016 = 0
    sum_of_p_given_2020 = 0
         
    for word in bow:
            sum_of_p_given_2016 += model['log p(w|y=2016)'][word] * bow[word]
            
    for word in bow:
            sum_of_p_given_2020 += model['log p(w|y=2020)'][word] * bow[word]
    
    retval['log p(y=2016|x)'] = model['log prior']['2016'] + sum_of_p_given_2016
    retval['log p(y=2020|x)'] = model['log prior']['2020'] + sum_of_p_given_2020
    
    if retval['log p(y=2016|x)'] >= retval['log p(y=2020|x)']:
        retval['predicted y'] = 2016
    else:
        retval['predicted y'] = 2020 
        
    return retval