import nltk
import sys

# for to remove punctuation using string.punctuation
import string

# for compute_idfs
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    test = 0
    
    if test == 1:
        files = load_files("corpus")
    else:
        # Check command-line arguments
        if len(sys.argv) != 2:
            sys.exit("Usage: python questions.py corpus")

        # Calculate IDF values across files
        files = load_files(sys.argv[1])
        
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # empty dictionary
    filecontents = dict()
    # use os to get the files and the paths
    import os
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            # don't forget the encoding
            with open(os.path.join(os.getcwd(),directory + os.sep + file), encoding="utf8") as f:
                # in the dict, map to the filename the string of the contents of the file
                filecontents[file] = f.read()
                
    return filecontents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # lowercase and convert to list
    # punctuation and stopwords
    try:
        punctuation = string.punctuation
    except:
        # https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktLanguageVars.word_tokenize
        # to prevent the error: "Resource punkt not found" you have to do
        nltk.download('punkt')
        punctuation = string.punctuation
    
    try:
        stopwords = nltk.corpus.stopwords.words("english")
    except:
        # to prevent the error: "Resource stopwords not found." you have to do
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words("english")

    
    # https://blog.finxter.com/how-to-filter-a-list-in-python/
    tokenized = [x for x in nltk.tokenize.word_tokenize(document.lower()) if x not in punctuation and x not in stopwords]
    
    return tokenized


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # create empty dict: can't init each word as 0, because we don't have the words yet
    wordidfs = dict()

    # for-loop for each file in documents
    for file in documents:
        # start with an empty set for this file
        wordslist = set()

        for word in documents[file]:
            if word not in wordslist:
                wordslist.add(word)
                # use try because the word might not be in the wordidfs dict yet
                try:
                    wordidfs[word] += 1
                except KeyError:
                    wordidfs[word] = 1

    # return the idf (natural logarithm of number of documents / number of documents in which the word appears)
    return {word: math.log(len(documents) / wordidfs[word]) for word in wordidfs}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # dict to hold scores for files: init at 0 (so we can += without try/except)
    scores = {filename:0 for filename in files}

    for word in query:
        # we get a list of words, only look at those
        if word in idfs:
            for filename in files:
                # count frequency of a word in the filecontents
                tf = files[filename].count(word)
                # calculate the idf of the word: frequency * idfs
                tf_idf = tf * idfs[word]
                # add the idf of the word to the dictionary
                scores[filename] += tf_idf

    # reverse sort the dictionary based on scores
    sorted_by_score = sorted([filename for filename in files], key=lambda x: scores[x], reverse=True)
    
    # only return the first n hits
    return sorted_by_score[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # empty dict (don't need to initialize with 0)
    scores = dict()
    
    # loop over keys (k=sentence) and values (v=words) of the items() in the sentences dict
    for k, v in sentences.items():
        # initialize score as 0 for each loop
        score = 0
        for word in query:
            if word in v:
                # the query-word is in the values of the sentence, so add the idfs[word] to the score
                score += idfs[word]

        # if score is not 0, we had a "hit" on this sentence
        if score != 0:
            # density = proportion of words in the sentence that are also words in the query
            density = sum([v.count(x) for x in query]) / len(v)
            # store the score and density in the scores dict
            scores[k] = (score, density)

    # we use density (x[1][1]) to break ties in idf ranking (x[1][0])
    sorted_by_score = [k for k, v in sorted(scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    # only return the first n hits
    return sorted_by_score[:n]


if __name__ == "__main__":
    main()
