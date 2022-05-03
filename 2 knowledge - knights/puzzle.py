from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Each character is either a knight or a knave.
puzzle_knowledge = And(
    Or(AKnight, AKnave),Not(And(AKnave, AKnight)),
    Or(BKnight, BKnave),Not(And(BKnave, BKnight)),
    Or(CKnight, CKnave),Not(And(CKnave, CKnight))
)

# A knight will always tell the truth:
# if a knight states a sentence, then that sentence is true.
# Conversely, a knave will always lie:
# if a knave states a sentence, then that sentence is false.

# Puzzle 0
knowledge0 = And(
    # the knowledge that a character is either knight or knave
    puzzle_knowledge,
    # A says "I am both a knight and a knave."
    # as AKnight he makes this statement (always true)
    Implication(AKnight, And(AKnight, AKnave)),
    # as AKnave he makes this statement (always false)
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
knowledge1 = And(
    puzzle_knowledge,
    # A says "We are both knaves."
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
    # B says nothing.
)

# Puzzle 2
knowledge2 = And(
    puzzle_knowledge,
    # A says "We are the same kind."
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    # B says "We are of different kinds."
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight))))
)

# Puzzle 3
knowledge3 = And(
    puzzle_knowledge,
    # A says either "I am a knight." or "I am a knave.", but you don't know which.
    Implication(AKnight, Or(AKnight, AKnave)),
    Implication(AKnave, Not(Or(AKnight, AKnave))),
    # B says "A said 'I am a knave'."
    # B says "C is a knave."
    Implication(BKnight, And(Implication(AKnight, AKnave), CKnave)),
    Implication(BKnave, Not(And(Implication(AKnight, AKnave), CKnave))),
    # C says "A is a knight."
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
