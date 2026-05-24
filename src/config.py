BOARD_SIZE = 15
CELL_SIZE = 45

WIN_LENGTH = 5


def is_winning_count(count):
    """
    Kiểm tra số quân liên tiếp đã đạt điều kiện thắng hay chưa.

    Dự án dùng luật caro đơn giản: một chuỗi có từ WIN_LENGTH quân trở lên
    được tính là thắng, bao gồm cả trường hợp hơn 5 quân liên tiếp.
    """
    return count >= WIN_LENGTH
