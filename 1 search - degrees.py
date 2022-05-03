import csv
import sys

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(f"people {row}")
            people[int(row["id"])] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {int(row["id"])}
            else:
                names[row["name"].lower()].add(int(row["id"]))

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(f"movies {row}")
            movies[int(row["id"])] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(f"stars {row}")
            try:
                people[int(row["person_id"])]["movies"].add(int(row["movie_id"]))
                movies[int(row["movie_id"])]["stars"].add(int(row["person_id"]))
            except KeyError:
                pass


def main():
    test = 1

    if test:
        import time
        # Load data from files into memory
        print("Loading data...")
        load_data("large")
        print("Data loaded.")
        
        print("---------- running check 4 ----------")
        source, target = "James Cagney", "John Cleese"
        
        print("---------- running check 4 NIEUW ----------")
        start_time = time.time()
        path = shortest_path_nieuw(person_id_for_name(source), person_id_for_name(target))
        print_check(path, source, target, correct_length=4)
        print("Dat duurde", format(time.time() - start_time, ".4f"), " seconden!")
        
        # print("---------- running check 4 OUD ----------")
        # start_time = time.time()
        # path = shortest_path_oud(person_id_for_name(source), person_id_for_name(target))
        # print_check(path, source, target, correct_length=4)
        # print("Dat duurde", format(time.time() - start_time, ".4f"), " seconden!")

    else:
        if len(sys.argv) > 2:
            sys.exit("Usage: python degrees.py [directory]")
        directory = sys.argv[1] if len(sys.argv) == 2 else "large"

        # Load data from files into memory
        print("Loading data...")
        load_data(directory)
        print("Data loaded.")

        source = person_id_for_name(input("Name: "))
        if source is None:
            sys.exit("Person not found.")
        target = person_id_for_name(input("Name: "))
        if target is None:
            sys.exit("Person not found.")

        path = shortest_path(source, target)

        if path is None:
            print("Not connected.")
        else:
            degrees = len(path)
            print(f"{degrees} degrees of separation.")
            path = [(None, source)] + path
            for i in range(degrees):
                person1 = people[path[i][1]]["name"]
                person2 = people[path[i + 1][1]]["name"]
                movie = movies[path[i + 1][0]]["title"]
                print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def shortest_path_oud(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    if source == target:
        return None # this would be weird, but I guess it could happen

    # in this case a BFS would work: always expand the shallowest node in the frontier
    # meaning, use a queuefrontier
    frontier = QueueFrontier()

    # we need a start node (each actor will be a node, the movie will be an action)
    # https://www.codecademy.com/learn/learn-data-structures-and-algorithms-with-python/modules/nodes/cheatsheet
    # Node comes from util.py: (state, parent, action)
    # now, we need to fill the frontier with the start node (the source)
    frontier.add(Node(source, None, None))

    # we need to keep track of the nodes we have visited (checked)
    # we initialize it before we start travelling
    visited_nodes = set()

    # we only need to keep track until we get to the target
    while True:
        # if there is no path, BFS will run until there are no more nodes
        # meaning the frontier is empty
        if frontier.empty():
            return None

        # remove the node we are checking (this removes the last entry on the frontier)
        node = frontier.remove()
        # add the node.state (the current person_id) to the visitied visited_nodes
        visited_nodes.add(node.state)

        # now, look at the actors for the current node
        # start by looking if the actor (the node.state) is the target
        if str(node.state) == str(target):
            # it is the target! so we're basically done, we just need to retrace our steps (back to the parents)
            path = []
            while node.parent is not None:
                # add the step to the path
                path.append((node.action,node.state))
                node = node.parent
            # we are done, so reverse our path and return it
            path.reverse()
            return path

        # now we should look at the movies for the current node
        # and get the actors/nodes connected to that
        for movie_id, person_id in neighbors_for_person(node.state):
            # we can skip the actors that we've checked before
            if person_id not in visited_nodes:
                # print(f"movie {movie_id} actor {person_id}")
                # from this person_id, build a node and add it to the frontier
                # node is the current node, so that is the parent for the new node
                frontier.add(Node(person_id, node, movie_id))
                
def shortest_path_nieuw(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    if source == target:
        return None # this would be weird, but I guess it could happen

    # in this case a BFS would work: always expand the shallowest node in the frontier
    # meaning, use a queuefrontier
    frontier = QueueFrontier()

    # we need a start node (each actor will be a node, the movie will be an action)
    # https://www.codecademy.com/learn/learn-data-structures-and-algorithms-with-python/modules/nodes/cheatsheet
    # Node comes from util.py: (state, parent, action)
    # now, we need to fill the frontier with the start node (the source)
    frontier.add(Node(source, None, None))

    # we need to keep track of the nodes we have visited (checked)
    # we initialize it before we start travelling
    visited_nodes = set()

    # we only need to keep track until we get to the target
    while True:
        # if there is no path, BFS will run until there are no more nodes
        # meaning the frontier is empty
        if frontier.empty():
            return None

        # remove the node we are checking (this removes the last entry on the frontier)
        node = frontier.remove()
        # add the node.state (the current person_id) to the visitied visited_nodes
        visited_nodes.add(node.state)

        # now we should look at the movies for the current node
        # and get the actors/nodes connected to that
        for movie_id, person_id in neighbors_for_person(node.state):
            # now, look at the actors for the current node
            # start by looking if the actor (the person_id) is the target
            if str(person_id) == str(target):
                # it is the target! so we're basically done, we just need to retrace our steps (back to the parents)
                # first, add the last step to the path
                path = [(movie_id, person_id)]
                # and then retrace
                while node.parent is not None:
                    # add the step to the path
                    path.append((node.action,person_id))
                    node = node.parent
                # we are done, so reverse our path and return it
                path.reverse()
                return path
            # we can skip the actors that we've checked before
            if person_id not in visited_nodes:
                # print(f"movie {movie_id} actor {person_id}")
                # from this person_id, build a node and add it to the frontier
                # node is the current node, so that is the parent for the new node
                frontier.add(Node(person_id, node, movie_id))
                

def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors

def check():
    # SMALL dataset checks
    load_data("small")
    # Check 1
    print("---------- runnning check 1 ----------")
    source, target = "Tom Cruise", "Tom Hanks"
    path = shortest_path(person_id_for_name(source), person_id_for_name(target))
    print_check(path, source, target, correct_length=2)
    # Check 2
    print("---------- runnning check 2 ----------")
    source, target = "Tom Hanks", "Emma Watson"
    path = shortest_path(person_id_for_name(source), person_id_for_name(target))
    print_check(path, source, target, correct_length=None)
    
    # LARGE dataset checks
    load_data("large")
    # Check 
    print("---------- runnning check 3 ----------")
    source, target = "Emma Watson","Robin Wright"
    path = shortest_path(person_id_for_name(source), person_id_for_name(target))
    print_check(path, source, target, correct_length=2)
    # Check 3
    print("---------- runnning check 4 ----------")
    source, target = "James Cagney", "John Cleese"
    path = shortest_path(person_id_for_name(source), person_id_for_name(target))
    print_check(path, source, target, correct_length=4)
    # Check 3
    print("---------- runnning check 5 ----------")
    source, target = "Quentin Tarantino", "Alan Arkin"
    path = shortest_path(person_id_for_name(source), person_id_for_name(target))
    print_check(path, source, target, correct_length=3)
    
def print_check(path, source, target, correct_length):
    if not path:
        if path == correct_length:
            print(f":) Correctly found no path between {source} and {target}\n")
    elif len(path) != correct_length:
        print(f":( Did not find the shortest path between {source} and {target}\n\
        --> path should be of length {correct_length}, but instead found path of length {len(path)}\n")
    else:
        print(f":) Correctly found shortest path between {source} and {target} of length {correct_length}\n")

if __name__ == "__main__":
    main()
