import numpy as np
import chess
import tensorflow as tf
import tensorflow.keras.models as models
import random
import sys

def legal_list(input_board):
    a = str(input_board.legal_moves)
    x = a.rfind('(')
    y = a.rfind(')')
    x += 1
    a = a[x:y]
    elements = a.split(',')
    a = [element.strip() for element in elements]
    b = []
    for i in a:
        b.append(str(input_board.push_san(i).uci()))
        input_board.pop()
    return b

# Define a function to convert FEN to a tensor
def board_to_tensor(input_board):

    # Convert board to FEN
    a = str(input_board.fen())
    x = a.find(' ')
    fen = a[0:x]
    
    # Define a mapping for chess pieces
    piece_mapping = {
        'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Black pawn
        'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Black knight
        'r': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Black rook
        'b': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Black bishop
        'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Black queen
        'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Black king
        'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # White pawn
        'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # White knight
        'R': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # White rook
        'B': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # White bishop
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # White queen
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # White king
        '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Empty square
    }

    tensor = np.zeros((8, 8, 13), dtype=np.int8)

    # Split the FEN into its components
    fen_parts = fen.split(' ')
    board = fen_parts[0]

    # Convert FEN to tensor
    row, col = 0, 0
    for char in board:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():  # Handle empty squares (digits represent the number of empty squares)
            col += int(char)
        else:
            tensor[row, col] = piece_mapping[char]
            col += 1

    tensor = tf.expand_dims(tensor, axis=0)
    return tensor

def square_to_uci(square_index):
    row = 7 - square_index // 8
    col = chr(ord('a') + square_index % 8)
    return col + str(row + 1)

def select_move(output_tensor, legal_list):
    output_array = output_tensor[0]
    osquare_unsorted = output_array[:64]
    dsquare_unsorted = output_array[64:]
    osquare_sorted = np.argsort(osquare_unsorted)[::-1]
    dsquare_sorted = np.argsort(dsquare_unsorted)[::-1]
    osquare = [index + 1 for index in osquare_sorted]
    dsquare = [index + 1 for index in dsquare_sorted]
    stop = False
    for i in osquare:
        if stop:
            break
        a = square_to_uci(i)
        for e in dsquare:
            b = square_to_uci(e)
            c = a + b
            if c in legal_list:
                stop = True
                return c
                break

def play_game(white_model, black_model, show, form):
    board = chess.Board()
    counter = 0
    uci_list = []
    while (not board.is_checkmate()) and (((not board.is_stalemate()) and (not board.is_fifty_moves())) and (not board.can_claim_threefold_repetition())):
        counter += 1
        if board.turn:
            legal_list1 = legal_list(board)
            input_tensor = board_to_tensor(board)
            output_tensor = white_model(input_tensor)
            move = select_move(output_tensor, legal_list1)
            if move == None:
                print("Warning: Move is NoneType")
                break
            if board.is_legal(chess.Move.from_uci(move)):
                a = board.push_uci(move)
                if form == 'uci':
                    board.pop()
                    b = board.san(a)
                    print(b)
                    board.push_uci(move)
            else:
                print("Warning: Illegal move")
                break
            last_move = True
            if form == 'uci':
                uci_list.append(move)
        elif not board.turn:
            if (not board.is_checkmate()) and (((not board.is_stalemate()) and (not board.is_fifty_moves())) and (not board.can_claim_threefold_repetition())):
                legal_list1 = legal_list(board)
                input_tensor = board_to_tensor(board)
                output_tensor = black_model(input_tensor)
                move = select_move(output_tensor, legal_list1)
                if move == None:
                    print("Warning: Move is NoneType")
                    break
                if board.is_legal(chess.Move.from_uci(move)):
                    a = board.push_uci(move)
                    if form == 'uci':
                        board.pop()
                        b = board.san(a)
                        print(b)
                        board.push_uci(move)
                else:
                    print("Warning: Illegal move")
                    break
                last_move = False
            else:
                break
        if show:
            if form == 'visual':
                print(" ")
                print("Move " + str(counter))
                print(board)
                print(" ")
                print("--------------------")
    if board.is_checkmate():
        checkmate = True
        if last_move:
            white_win = True
        else:
            white_win = False
        return checkmate, white_win
    else:
        return False, False

def make_player_model():
    model = tf.keras.Sequential()
    
    # Flatten the input tensor
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 13)))
    
    model.add(tf.keras.layers.Dense(832, activation='tanh'))
    model.add(tf.keras.layers.Dense(700, activation='linear')) # Additional hidden layer
    model.add(tf.keras.layers.Dense(600, activation='linear'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(500, activation='linear'))
    model.add(tf.keras.layers.Dense(400, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(250, activation='tanh'))
    model.add(tf.keras.layers.Dense(200, activation='linear')) # Additional hidden layer
    model.add(tf.keras.layers.Dense(128, activation='linear'))  # Use softmax for multi-class classification
    return model

def tweak_weights(model, scale, one_in_chance):
    tweaked_weights = []
    one_in_chance -= 1
    
    for layer_weights in model.get_weights():
        a = random.randint(0, one_in_chance)
        if a == 0:
            
            # Generate random values with the same shape as the layer's weights
            random_values = np.random.normal(loc=0.0, scale=scale, size=layer_weights.shape)
            
            # Add the random values to the layer's weights
            tweaked_layer_weights = layer_weights + random_values
            tweaked_weights.append(tweaked_layer_weights)
        else:
            tweaked_weights.append(layer_weights)
    
    return tweaked_weights

def train(white_model, black_model):

    #Train white

    complete = False
    while complete == False:

        candidate = white_model
        try:
            tweaked_weights = tweak_weights(white_model, 0.01, 7)
        except:
            print("tweak_weights() Error")
            break
        candidate.set_weights(tweaked_weights)
        checkmate, white_win = play_game(candidate, black_model, False, False)
        if checkmate:
            if white_win:
                white_model = candidate
                print("White wins!")
                complete = True
            else:
                print("White loses!")
        else:
            print("White draws")

    white_model.save_weights(r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\chessbot\white_model.h5")
    print("White model saved")

    # Train black
    
    complete = False
    while complete == False:

        candidate = black_model
        try:
            tweaked_weights = tweak_weights(black_model, 0.01, 7)
        except:
            print("tweak_weights() Error")
            break
        candidate.set_weights(tweaked_weights)
        checkmate, white_win = play_game(white_model, black_model, False, False)
        if checkmate:
            if not white_win:
                black_model = candidate
                print("Black wins!")
                complete = True
            else:
                print("Black loses!")
        else:
            print("Black draws")

    black_model.save_weights(r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\chessbot\black_model.h5")
    print("Black model saved")

white_model = make_player_model()
black_model = make_player_model()

scratch = input("Load models? y/n: ")
if scratch == "y":
    sample_input = tf.keras.Input(shape=(8, 8, 13))
    white_model(sample_input)
    white_model.load_weights(r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\chessbot\white_model.h5")
    black_model(sample_input)
    black_model.load_weights(r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\chessbot\black_model.h5")

if_train = input("Train the model? y/n: ")
if if_train == "y":
    epochs = int(input("Epochs: "))
    for i in range(epochs):
        train(white_model, black_model)
    
play_game(white_model, black_model, True, 'uci')
