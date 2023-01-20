import pickle
objects = []
with (open("ptb.vocab.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break



print(openfile)
