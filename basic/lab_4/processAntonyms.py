if __name__ == '__main__':
    f = open("antonyms", "r")
    text = f.read()
    f.close()
    tokens = text.split("\n")
    for t in tokens:
        words = t.split("–")
        if len(words) == 2:
            alts = words[1].split(",")
            for a in alts:
                print("[\"" + words[0].strip() + "\", \"" + a.strip() + "\"],")
