def distance_a(path1, path2):
    total = 0
    diverged = 0
    for i in range(min(len(path1), len(path2))):  # finds depth of divergence
        if path1[i] != path2[i]:
            total += i
            diverged = True
            break
    if not diverged:
        return len(path1)
    return total


def distance_b(path1, path2):
    total = 0
    diverged = 0
    for i in range(min(len(path1), len(path2))):  # finds depth of divergence
        if path1[i] != path2[i]:
            total += i
            diverged = i
            break
    if diverged != 0:
        for j in range(diverged, len(path1)):
            total -= j
        for k in range(diverged, len(path2)):
            total -= k
    if not diverged:
        return len(path1)
    return total


def distance_c(path1, path2):
    total = 0
    diverged = 0
    for i in range(min(len(path1), len(path2))):  # finds depth of divergence
        if path1[i] != path2[i]:
            total += i
            diverged = i
            break
    if diverged != 0:
        for j in range(1, len(path1)):
            total -= j
        for k in range(1, len(path2)):
            total -= k
    if not diverged:
        return len(path1)
    return total


def get_sibling_distance(path1, path2):
    if len(path1) == len(path2):
        l = len(path1)
        if path1[l-1] != path2[l-1] and path1[l-2] == path2[l-2]:
            return 1
    return 0


def get_distance(word1, word2, dictionary):
    a = distance_a(dictionary[word1], dictionary[word2])
    b = distance_b(dictionary[word1], dictionary[word2])
    c = distance_c(dictionary[word1], dictionary[word2])
    d = get_sibling_distance(dictionary[word1], dictionary[word2])

    return a, b, c, d

