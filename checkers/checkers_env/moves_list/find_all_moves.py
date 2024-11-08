import pickle

# => 196 standard non-capturing moves:
# => 144 capture 1st grade (ONE PIECE / KING)
# => 256 capture 2st grade (TWO PIECE / KING)
# => 368 capture 3nd grade (THREE PIECES / KINGS)
# => 416 capture 4rd grade (FOUR PIECES / KINGS)
# => 480 capture 5th grade (FIVE PIECES / KINGS)
# => 416 capture 6th grade (SIX PIECES / KINGS)
# => 288 capture 7th grade (SEVEN PIECES / KINGS)
# => 192 capture 8th grade (EIGHT PIECES / KINGS)
# => 96 capture 8th grade (NINE PIECES / KINGS)
# => 0 capture 8th grade (TEN PIECES / KINGS)

# => TOTAL MOVES == 2852
# => MAX CAPTURE LENGTH == 9


def check_valid(captures_):
    for element in captures_:
        if captures_.count(element) > 1:
            return False
    return True


def no_capture_moves():
    move_list_n = []
    directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    for x1 in range(8):
        for y1 in range(8):
            for direction in directions:
                x2 = x1 + direction[0]
                y2 = y1 + direction[1]
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    transition = ((x1, y1), (x2, y2))
                    if transition not in move_list_n:
                        move_list_n.append(transition)
    return move_list_n, None


def capture_1():
    move_list_n = []
    captures_list_n = []
    directions_captures = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction in directions:
                x2 = x1 + direction[0]
                y2 = y1 + direction[1]
                x1_c = x1 + (direction[0] // 2)
                y1_c = y1 + (direction[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    transition = ((x1, y1), (x2, y2))
                    captures = ((x1_c, y1_c))
                    if transition not in move_list_n:
                        move_list_n.append(transition)
                        captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_2():
    move_list_n = []
    captures_list_n = []
    directions_captures = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            transition = ((x1, y1), (x2, y2), (x3, y3))
                            captures = ((x1_c, y1_c), (x2_c, y2_c))
                            if check_valid(captures) and transition not in move_list_n:
                                move_list_n.append(transition)
                                captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_3():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                                    captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c))
                                    if check_valid(captures) and transition not in move_list_n:
                                        move_list_n.append(transition)
                                        captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_4():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5))
                                            captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c), (x4_c, y4_c))
                                            if check_valid(captures) and transition not in move_list_n:
                                                move_list_n.append(transition)
                                                captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_5():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6))
                                                    captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                (x4_c, y4_c), (x5_c, y5_c))
                                                    if check_valid(captures) and transition not in move_list_n:
                                                        move_list_n.append(transition)
                                                        captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_6():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    for direction6 in directions:
                                                        x7 = x6 + direction6[0]
                                                        y7 = y6 + direction6[1]
                                                        x6_c = x6 + (direction6[0] // 2)
                                                        y6_c = y6 + (direction6[1] // 2)
                                                        if 0 <= x7 < 8 and 0 <= y7 < 8:
                                                            transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7))
                                                            captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                        (x4_c, y4_c), (x5_c, y5_c), (x6_c, y6_c))
                                                            if check_valid(captures) and transition not in move_list_n:
                                                                move_list_n.append(transition)
                                                                captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_7():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    for direction6 in directions:
                                                        x7 = x6 + direction6[0]
                                                        y7 = y6 + direction6[1]
                                                        x6_c = x6 + (direction6[0] // 2)
                                                        y6_c = y6 + (direction6[1] // 2)
                                                        if 0 <= x7 < 8 and 0 <= y7 < 8:
                                                                for direction7 in directions:
                                                                    x8 = x7 + direction7[0]
                                                                    y8 = y7 + direction7[1]
                                                                    x7_c = x7 + (direction7[0] // 2)
                                                                    y7_c = y7 + (direction7[1] // 2)
                                                                    if 0 <= x8 < 8 and 0 <= y8 < 8:
                                                                        transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8))
                                                                        captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                                        (x4_c, y4_c), (x5_c, y5_c), (x6_c, y6_c), (x7_c, y7_c))
                                                                        if check_valid(captures) and transition not in move_list_n:
                                                                            move_list_n.append(transition)
                                                                            captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_8():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    for direction6 in directions:
                                                        x7 = x6 + direction6[0]
                                                        y7 = y6 + direction6[1]
                                                        x6_c = x6 + (direction6[0] // 2)
                                                        y6_c = y6 + (direction6[1] // 2)
                                                        if 0 <= x7 < 8 and 0 <= y7 < 8:
                                                            for direction7 in directions:
                                                                x8 = x7 + direction7[0]
                                                                y8 = y7 + direction7[1]
                                                                x7_c = x7 + (direction7[0] // 2)
                                                                y7_c = y7 + (direction7[1] // 2)
                                                                if 0 <= x8 < 8 and 0 <= y8 < 8:
                                                                    for direction8 in directions:
                                                                        x9 = x8 + direction8[0]
                                                                        y9 = y8 + direction8[1]
                                                                        x8_c = x8 + (direction8[0] // 2)
                                                                        y8_c = y8 + (direction8[1] // 2)
                                                                        if 0 <= x9 < 8 and 0 <= y9 < 8:
                                                                            transition = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8), (x9, y9))
                                                                            captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                                            (x4_c, y4_c), (x5_c, y5_c), (x6_c, y6_c), (x7_c, y7_c), (x8_c, y8_c))
                                                                            if check_valid(captures) and transition not in move_list_n:
                                                                                move_list_n.append(transition)
                                                                                captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_9():
    move_list_n = []
    captures_list_n = []
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    for direction6 in directions:
                                                        x7 = x6 + direction6[0]
                                                        y7 = y6 + direction6[1]
                                                        x6_c = x6 + (direction6[0] // 2)
                                                        y6_c = y6 + (direction6[1] // 2)
                                                        if 0 <= x7 < 8 and 0 <= y7 < 8:
                                                            for direction7 in directions:
                                                                x8 = x7 + direction7[0]
                                                                y8 = y7 + direction7[1]
                                                                x7_c = x7 + (direction7[0] // 2)
                                                                y7_c = y7 + (direction7[1] // 2)
                                                                if 0 <= x8 < 8 and 0 <= y8 < 8:
                                                                    for direction8 in directions:
                                                                        x9 = x8 + direction8[0]
                                                                        y9 = y8 + direction8[1]
                                                                        x8_c = x8 + (direction8[0] // 2)
                                                                        y8_c = y8 + (direction8[1] // 2)
                                                                        if 0 <= x9 < 8 and 0 <= y9 < 8:
                                                                            for direction9 in directions:
                                                                                x10 = x9 + direction9[0]
                                                                                y10 = y9 + direction9[1]
                                                                                x9_c = x9 + (direction9[0] // 2)
                                                                                y9_c = y9 + (direction9[1] // 2)
                                                                                if 0 <= x10 < 8 and 0 <= y10 < 8:
                                                                                    transition = ((x1, y1), (x2, y2), (x3, y3),
                                                                                                  (x4, y4), (x5, y5), (x6, y6),
                                                                                                  (x7, y7), (x8, y8), (x9, y9),
                                                                                                  (x10, y10))
                                                                                    captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                                                (x4_c, y4_c), (x5_c, y5_c), (x6_c, y6_c),
                                                                                                (x7_c, y7_c), (x8_c, y8_c), (x9_c, y9_c))
                                                                                    if check_valid(captures) and transition not in move_list_n:
                                                                                        move_list_n.append(transition)
                                                                                        captures_list_n.append(captures)
    return move_list_n, captures_list_n


def capture_10():
    move_list_n = set()
    captures_list_n = set()
    directions = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for x1 in range(8):
        for y1 in range(8):
            for direction1 in directions:
                x2 = x1 + direction1[0]
                y2 = y1 + direction1[1]
                x1_c = x1 + (direction1[0] // 2)
                y1_c = y1 + (direction1[1] // 2)
                if 0 <= x2 < 8 and 0 <= y2 < 8:
                    for direction2 in directions:
                        x3 = x2 + direction2[0]
                        y3 = y2 + direction2[1]
                        x2_c = x2 + (direction2[0] // 2)
                        y2_c = y2 + (direction2[1] // 2)
                        if 0 <= x3 < 8 and 0 <= y3 < 8:
                            for direction3 in directions:
                                x4 = x3 + direction3[0]
                                y4 = y3 + direction3[1]
                                x3_c = x3 + (direction3[0] // 2)
                                y3_c = y3 + (direction3[1] // 2)
                                if 0 <= x4 < 8 and 0 <= y4 < 8:
                                    for direction4 in directions:
                                        x5 = x4 + direction4[0]
                                        y5 = y4 + direction4[1]
                                        x4_c = x4 + (direction4[0] // 2)
                                        y4_c = y4 + (direction4[1] // 2)
                                        if 0 <= x5 < 8 and 0 <= y5 < 8:
                                            for direction5 in directions:
                                                x6 = x5 + direction5[0]
                                                y6 = y5 + direction5[1]
                                                x5_c = x5 + (direction5[0] // 2)
                                                y5_c = y5 + (direction5[1] // 2)
                                                if 0 <= x6 < 8 and 0 <= y6 < 8:
                                                    for direction6 in directions:
                                                        x7 = x6 + direction6[0]
                                                        y7 = y6 + direction6[1]
                                                        x6_c = x6 + (direction6[0] // 2)
                                                        y6_c = y6 + (direction6[1] // 2)
                                                        if 0 <= x7 < 8 and 0 <= y7 < 8:
                                                            for direction7 in directions:
                                                                x8 = x7 + direction7[0]
                                                                y8 = y7 + direction7[1]
                                                                x7_c = x7 + (direction7[0] // 2)
                                                                y7_c = y7 + (direction7[1] // 2)
                                                                if 0 <= x8 < 8 and 0 <= y8 < 8:
                                                                    for direction8 in directions:
                                                                        x9 = x8 + direction8[0]
                                                                        y9 = y8 + direction8[1]
                                                                        x8_c = x8 + (direction8[0] // 2)
                                                                        y8_c = y8 + (direction8[1] // 2)
                                                                        if 0 <= x9 < 8 and 0 <= y9 < 8:
                                                                            for direction9 in directions:
                                                                                x10 = x9 + direction9[0]
                                                                                y10 = y9 + direction9[1]
                                                                                x9_c = x9 + (direction9[0] // 2)
                                                                                y9_c = y9 + (direction9[1] // 2)
                                                                                if 0 <= x10 < 8 and 0 <= y10 < 8:
                                                                                    for direction10 in directions:
                                                                                        x11 = x10 + direction10[0]
                                                                                        y11 = y10 + direction10[1]
                                                                                        x10_c = x10 + (direction10[0] // 2)
                                                                                        y10_c = y10 + (direction10[1] // 2)
                                                                                        if 0 <= x11 < 8 and 0 <= y11 < 8:
                                                                                            transition = ((x1, y1), (x2, y2), (x3, y3),
                                                                                                          (x4, y4), (x5, y5), (x6, y6),
                                                                                                          (x7, y7), (x8, y8), (x9, y9),
                                                                                                          (x10, y10), (x11, y11))
                                                                                            captures = ((x1_c, y1_c), (x2_c, y2_c), (x3_c, y3_c),
                                                                                                        (x4_c, y4_c), (x5_c, y5_c), (x6_c, y6_c),
                                                                                                        (x7_c, y7_c), (x8_c, y8_c), (x9_c, y9_c), (x10_c, y10_c))
                                                                                            if check_valid(captures):
                                                                                                move_list_n.add(transition)
                                                                                                captures_list_n.add(captures)
    return move_list_n, captures_list_n


moves = []
captures = []

# NO CAPTURES
move_list, _ = no_capture_moves()
for move in move_list:
    moves.append(move)
    captures.append(())

# CAPTURE 1
move_list, capture_list = capture_1()
for move in move_list:
    moves.append(move)

for capture in capture_list:
    captures.append(capture)

# CAPTURE 2
move_list, capture_list = capture_2()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 3
move_list, capture_list = capture_3()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 4
move_list, capture_list = capture_4()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 5
move_list, capture_list = capture_5()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 6
move_list, capture_list = capture_6()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 7
move_list, capture_list = capture_7()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 8
move_list, capture_list = capture_8()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

# CAPTURE 9
move_list, capture_list = capture_9()
for move in move_list:
    moves.append(move)
for capture in capture_list:
    captures.append(capture)

print(len(moves))
print(len(captures))
moves = tuple(moves)
captures = tuple(captures)

with open("moves_list.pkl", "wb") as f:
    pickle.dump(moves, f)

with open("captures_list.pkl", "wb") as f:
    pickle.dump(captures, f)
