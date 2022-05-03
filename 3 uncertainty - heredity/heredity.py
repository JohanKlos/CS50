import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
#     if len(sys.argv) != 2:
#         sys.exit("Usage: python heredity.py data.csv")
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = ".\\data\\family2.csv"
    print(arg)
    people = load_data(arg)
    """
    This is what your project will compute:
    for each person, your AI will calculate the probability distribution over
    how many of copies of the gene they have, as well as whether they have
    the trait or not. probabilities["Harry"]["gene"][1], for example,
    will be the probability that Harry has 1 copy of the gene, and
    probabilities["Lily"]["trait"][False] will be the probability that
    Lily does not exhibit the trait.
    ___________________________________________________________________________________
    We can calculate a conditional probability by summing up all of the joint
    probabilities that satisfy the evidence, and then normalize those probabilities
    so that they each sum to 1.
    """
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # this part checks for every person in names
        # if a person has a trait
        # AND
        # if that person is not in the powerset list of have_trait

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(a) for a in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # this function computes the probability for a certain "world"
    # all possible worlds are computed (does not have to be determined in this function)
    
    joint_probability = 1

    for person in people:
        # initial probability
        prob_person = 1

        # set some variables for the person in question
        if person in one_gene:
            genes_person = 1
        elif person in two_genes:
            genes_person = 2
        else:
            genes_person = 0
            
        if person in have_trait:
            trait_person = 1
        else:
            trait_person = 0
        
        
        father = people[person]["father"]
        mother = people[person]["mother"]
        
        if father == None and mother == None:
            # no parents listed: use PROBS["gene"] for probability of number of the gene
            # print(person,"has no known parents")
            prob_person *= PROBS["gene"][genes_person]
        else:
            # parents listed, so see the number of genes they have
            # father
            if father in two_genes:
                prob_father = 1 - PROBS['mutation']
            elif father in one_gene:
                prob_father = 0.5
            else:
                prob_father = PROBS['mutation']
                
            # print(father,prob_father)
            
            # mother
            if mother in two_genes:
                prob_mother = 1 - PROBS['mutation']
            elif mother in one_gene:
                prob_mother = 0.5
            else:
                prob_mother = PROBS['mutation']
            
            # print(mother,prob_mother)
            
            if genes_person == 2:
                prob_person *= prob_father * prob_mother
            elif genes_person == 1:
                prob_person *= (1 - prob_mother) * prob_father + (1 - prob_father) * prob_mother
            else:
                prob_person *= (1 - prob_mother) * (1 - prob_father)
            
        # compute the probability that a person does or does not have a particular trait
        prob_person *= PROBS['trait'][genes_person][trait_person]

        joint_probability *= prob_person
        
        # print(person,round(joint_probability,6))
        
    return joint_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # gene probility
        if person in one_gene:
            genes_person = 1
        elif person in two_genes:
            genes_person = 2     
        else:
            genes_person = 0

        probabilities[person]["gene"][genes_person] = probabilities[person]["gene"][genes_person] + p

        # trait probability
        if person in have_trait:
            trait_person = True
        else:
            trait_person = False

        probabilities[person]["trait"][trait_person] = probabilities[person]["trait"][trait_person] + p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # to normalize both gene and trait, do the following:
    # iterate over all people in probabilities
    # per person: sum up all the values (.values())
    # per person: divide each of the elements (.items()) by the relevant sum-total
    
    # Iterate over all people:
    for person in probabilities:
        # Normalise each distribution to 1
        # by dividing the person's values by the total probability for each distribution
        # do this for both 'gene'
        probabilities[person]['gene'] = { genes: ( prob / sum(probabilities[person]['gene'].values()) ) for genes, prob in probabilities[person]['gene'].items() }
        # and for 'trait'
        probabilities[person]['trait'] = { trait: ( prob / sum(probabilities[person]['trait'].values()) ) for trait, prob in probabilities[person]['trait'].items() }
    


if __name__ == "__main__":
    main()
