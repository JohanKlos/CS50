import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# komt Grondslagen toch nog van pas!
# remember recursion, it can refer to itself for repetition
NONTERMINALS = """
S -> NP VP | S Conj S | S P S
NP -> N | Det AdjN | NP PNP
AdjN -> N | Adj AdjN
PNP -> P NP
VP -> V | V NP | VP PNP | VP Adv | Adv VP | VP Conj VP
"""

"""
 : Holmes sat.
 : Holmes lit a pipe.
 : We arrived the day before Thursday.
 : Holmes sat in the red armchair and he chuckled.
 : My companion smiled an enigmatical smile.
 : Holmes chuckled to himself.
 : She never said a word until we were at the door here.
 : Holmes sat down and lit his pipe.
 : I had a country walk on Thursday and came home in a dreadful mess.
 : I had a little moist red paint in the palm of my hand.
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # create empty list
    words = []
    # make sentence lowercase and then split per space
    
    try:
        words = nltk.word_tokenize(sentence.lower())
    except:
        # https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktLanguageVars.word_tokenize
        # to prevent the error: "Resource punkt not found" you have to do
        nltk.download('punkt')
        words = nltk.word_tokenize(sentence.lower())
        
    
    # remove all words that does not contain at least one alphabetic char
    results = []
    for word in words:
        # if we just do isalpha() then words like "o'clock" are excluded
        # so use the any-for-in statement
        if any(c.isalpha() for c in word):
            results.append(word)
    
    return results


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # create empty list
    chunks = []
    
    # return a list of nltk.tree objects, where each element has the label NP
    # https://www.nltk.org/_modules/nltk/tree.html
    
    # we have to make a new subtree when an NP is found
    # recursion is key, so make a function for it
    def search_np(subtree):
        # search subtrees using recursion
        for sub in subtree.subtrees():
            # skip the current subtree
            if sub == subtree:
                continue    
            # search for NP
            if sub.label() == "NP":
                return True
            # recursion!
            if search_np(sub):
                return True
            # there was no return yet, so return False
        return False

    for sub in tree.subtrees():
        # for all subtrees labeled NP
        if sub.label() == "NP":
            # that does not contain any NP subtree
            if not search_np(sub):
                # append to the np chunks list
                chunks.append(sub)

    return chunks
    
    
if __name__ == "__main__":
    main()
