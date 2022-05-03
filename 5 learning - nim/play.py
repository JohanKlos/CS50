from nim import train, play

ai = train(10000)
play(ai)

againagainagain = input("Want to play another game? (y/n, default: y) ")
while againagainagain != "n":
    play(ai)
    againagainagain = input("Want to play another game? (y/n, default: y) ")
    