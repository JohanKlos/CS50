"""
https://elo.beta4all.nl/mod/assign/view.php?id=4603&forceview=1

method 1: Markov Chain
================================
Markov Chain, where each page represents a state, and each page has a transition
model that chooses among its links at random. At each time step, the state switches
to one of the pages linked to by the current state.

By sampling states randomly from the Markov Chain, we can get an estimate for each
page’s PageRank. We can start by choosing a page at random, then keep following
links at random, keeping track of how many times we’ve visited each page.
After we’ve gathered all of our samples (based on a number we choose in advance),
the proportion of the time we were on each page might be an estimate for that
page’s rank.

To ensure we can always get to somewhere else in the corpus of web pages, we’ll
introduce to our model a damping factor d. With probability d (where d is usually
set around 0.85), the random surfer will choose from one of the links on the
current page at random. But otherwise (with probability 1 - d), the random surfer
chooses one out of all of the pages in the corpus at random (including the one
they are currently on).

Our random surfer now starts by choosing a page at random, and then, for each
additional sample we’d like to generate, chooses a link from the current page at
random with probability d, and chooses any page at random with probability 1 - d.
If we keep track of how many times each page has shown up as a sample, we can
treat the proportion of states that were on a given page as its PageRank.

method 2: Iterative Algorithm
================================
We can also define a page’s PageRank using a recursive mathematical expression.
Let PR(p) be the PageRank of a given page p: the probability that a random surfer
ends up on that page. How do we define PR(p)?
Well, we know there are two ways that a random surfer could end up on the page:

1. With probability 1 - d, the surfer chose a page at random and ended up on page p.
2. With probability d, the surfer followed a link from a page i to page p.

The first condition is fairly straightforward to express mathematically:
it’s 1 - d divided by N, where N is the total number of pages across the entire corpus.
This is because the 1 - d probability of choosing a page at random is split evenly
among all N possible pages.

For the second condition, we need to consider each possible page i that links to page p.
For each of those incoming pages, let NumLinks(i) be the number of links on page i.
Each page i that links to p has its own PageRank, PR(i), representing the probability
that we are on page i at any given time. And since from page i we travel to any of
that page’s links with equal probability, we divide PR(i) by the number of links
NumLinks(i) to get the probability that we were on page i and chose the link to page p.

How would we go about calculating PageRank values for each page, then?
We can do so via iteration: start by assuming the PageRank of every page is 1 / N
(i.e., equally likely to be on any page). Then, use the above formula to calculate
new PageRank values for each page, based on the previous PageRank values.
If we keep repeating this process, calculating a new set of PageRank values for
each page based on the previous set of PageRank values, eventually the PageRank
values will converge (i.e., not change by more than a small threshold with each iteration).
"""

import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        arg = "corpus0"
    else:
        arg = sys.argv[1]
    corpus = crawl(arg)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    For example, if the corpus were
    {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}},
    the page was "1.html", and the damping_factor was 0.85,
    then the output of transition_model should be
    {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}.

    This is because with probability 0.85, we choose randomly to go from
    page 1 to either page 2 or page 3
    (so each of page 2 or page 3 has probability 0.425 to start),
    but every page gets an additional 0.05 because with probability 0.15
    we choose randomly among all of the pages in our world (0.15 / 3).
    """
    # print("links",corpus[page])

    # Probability of picking any page in our world at random:
    random_prob = (1 - damping_factor) / len(corpus)
    # print("prob of random page: ", round(random_prob,3))

    # Probability of picking a link from the page we are at:
    if len(corpus[page]) > 0:
        link_prob = damping_factor / len(corpus[page])
    # print(f"{page} links to {len(corpus[page])} pages. Prob per linked page: {round(link_prob,3)}")

    # keys (page_name below) in our dict need to be initialised to 0, so we can add to it
    # https://stackoverflow.com/questions/521674/initializing-a-list-to-a-known-number-of-elements-in-python
    # verts = [0 for x in range(1000)]
    prob = {page_name : 0 for page_name in corpus}

    # no outgoing links so return equal probability for all pages in corpus:
    if len(corpus[page]) == 0:
        for page_name in corpus:
            prob[page_name] = 1 / len(corpus)
        return prob

    # Add probabilities to our dictionary for the linked pages:
    for page_name in corpus:
        prob[page_name] += random_prob
        if page_name in corpus[page]:
            prob[page_name] += link_prob

    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # we want to return a dictionary, so start by initializing it (similar to transition_model)
    dict_prank = {page_name : 0 for page_name in corpus}
    # to calculate the probability of all samples, we need to keep track of the number of visits
    dict_visited = {page_name : 0 for page_name in corpus}

    # assert to make sure the transition_model functions properly
    assert (transition_model({"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}},"1.html",0.85) == {'1.html': 0.05000000000000001, '2.html': 0.475, '3.html': 0.475}), "transition_model wrong"

    # The first sample should be generated by choosing from a page at random.
    # because it's a dictionary, we can use random.choice(list(dict))
    # https://pynative.com/python-random-choice/
    picked_page = random.choice(list(dict_prank))

    # n = 10 # just for testing
    # we need to do n samples, starting from 0, that gives range(0, n-1)
    for i in range(0, n-1):
        # print(f"187: {picked_page}")
        tmodel = transition_model(corpus, picked_page, damping_factor)
        for page in tmodel:
            dict_prank[page] += tmodel[page]
            dict_visited[page] += 1

        # You will likely want to pass the previous sample into your transition_model
        # function, along with the corpus and the damping_factor, to get the probabilities
        # for the next sample.

        # the next choice should be made with the probability in mind
        # https://www.geeksforgeeks.org/how-to-get-weighted-random-choice-in-python/
        # https://stackoverflow.com/questions/40927221/how-to-choose-keys-from-a-python-dictionary-based-on-weighted-probability
        # then, pick a random page with the weights in mind
        picked_page = random.choices(list(tmodel.keys()),weights=tmodel.values())[0]

    # The return value of the function should be a Python dictionary with
    # one key for each page in the corpus. Each key should be mapped to a value
    # representing that page’s estimated PageRank (i.e., the proportion of all the
    # samples that corresponded to that page).
    # after we've done all the samples, we need to divide the sum by the number
    # of times we visited each page
    val_test = 0
    for page in dict_visited:
        if dict_visited[page] > 0:
            # print(f"{page} has a total probability of {dict_prank[page]} over {dict_visited[page]} visits")
            # print(dict_prank[page] / dict_visited[page])
            dict_prank[page] = dict_prank[page] / dict_visited[page]
            val_test += dict_prank[page]

    # print(f"val_test : {val_test}")
    # The values in this dictionary should sum to 1.
    assert (round(val_test,8) == 1), "sample_pagerank total sum of values of dict_prank needs to be 1 in total"

    return dict_prank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # variables we will keep using in this function
    number_pages = len(corpus)
    d = damping_factor

    # start by assuming the PageRank of every page is 1 / N (i.e., equally likely to be on any page).
    # begin by assigning each page a rank of 1 / total number of pages in the corpus
    # this is the first part (with 4 pages in our world, it'll be 0.25 for each page)
    dict_pagerank = {p : 1/number_pages for p in corpus}
    dict_newpagerank = {p : 0 for p in corpus}

    # first condition: 1 - d divided by N, where N is the total number of pages across the entire corpus.
    rand_page = ( 1 - d ) / number_pages

    # repeatedly calculate a new set of PageRank values for each page based on
    # the previous set of PageRank values, eventually the PageRank values will converge
    # so we repeat until no PageRank value changes by more than 0.001
    # between iterations: keep track of the maximum change in each iteration.
    max_change = 1/number_pages # just the initialization, if this is below the threshold we are already done
    while max_change > 0.001:
        # initialize for each loop
        max_change = 0
    # for j in range(0,10):
        new_pagerank = 0
        for p in corpus:
            probability = 0
            for i in corpus:
                # A page that has no links at all should be interpreted as having one link
                # for every page in the corpus (including itself).
                if len(corpus[i]) == 0:
                    probability += dict_pagerank[i] * (1/number_pages)

                # consider each possible page i that links to page p
                elif p in corpus[i]:
                    # For each of those incoming pages, let NumLinks(i) be the number of links on page i.
                    probability += dict_pagerank[i] / len(corpus[i])

            # get the new pagerank
            new_pagerank = rand_page + (d * probability)
            dict_newpagerank[p] = new_pagerank

        # we need to normalize the ranks (otherwise it won't ever amount to 1)
        # we do that by getting the sum of all the new_pageranks
        normalization_sum = sum(dict_newpagerank.values())
        # and divide the new_pagerank of this page by that sum
        new_pageranks = {p: (prank / normalization_sum) for p, prank in dict_newpagerank.items()}

        # because the while loop only stops when max_change <= 0.001, calculate that for each rank
       # Find max change in page rank:
        for p in corpus:
            change = abs(dict_pagerank[p] - dict_newpagerank[p])
            if change > max_change:
                max_change = change

        # make copy so we can iterate over the new ranks
        dict_pagerank = dict_newpagerank.copy()


    # The return value of the function should be a Python dictionary with one key
    # for each page in the corpus.
    # Each key should be mapped to a value representing that page’s PageRank.
    # The values in this dictionary should sum to 1.
    assert (round(sum(dict_pagerank.values()), 8) == 1), "iterate_pagerank total sum of values of dict_output needs to be 1.0"

    return dict_pagerank

if __name__ == "__main__":
    main()
