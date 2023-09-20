MORSE = {'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.', 'g': '--.', 'h': '....',
         'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
         'm': '--', 'n': '-.', 'o': '---', 'p': '.--.',
         'q': '--.-', 'r': '.-.', 's': '...', 't': '-',
         'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-', 'y': '-.--', 'z': '--..', ' ': '\n'}

CMDS = {'write': 'w', 'read': 'read', 'execute': 'x'}


def task_1() -> None:
    try:
        print(''.join([MORSE[i] for i in input("Введите строку для задания 1:\n").lower()]))
    except KeyError:
        print("Введены некорректные данные!")


def task_2():
    names, n = set(), int(input("Введите данные для задания 2:\n"))
    for i in range(n):
        s, j = input(), 1
        duplicate = s
        while True:
            if duplicate not in names:
                names.add(duplicate)
                print("OK") if duplicate == s else print(duplicate)
                break
            else:
                duplicate = s + str(i)
                j += 1


def task_3():
    n, m = int(input()), dict()
    for _ in range(n):
        try:
            file_privileges = input().split(' ')
            filename, privileges = file_privileges[0], file_privileges[1:]
        except IndexError:
            print("Неверные входные значения!")
            continue
        m[filename] = privileges

    n = int(input())
    for _ in range(n):
        file_operation = input().split(' ')
        try:
            operation = file_operation[0]
            filename = file_operation[1]
        except IndexError:
            print("Неверные входные значения!")
            continue

        lil_operation = CMDS[operation]
        print("OK") if lil_operation in m[filename] else print("Access denied")
