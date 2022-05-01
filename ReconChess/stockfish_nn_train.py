import re

import chess
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from eval_engine_stockfish_nn import SparseBatch
import pandas as pd
import torch.optim as optim


class BasicNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(832, 832)
        self.fc2 = nn.Linear(832, 416)
        self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(208, 104)
        self.fc5 = nn.Linear(104, 104)
        self.fc6 = nn.Linear(104, 104)
        self.fc7 = nn.Linear(104, 104)
        self.fc8 = nn.Linear(104, 1)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Dropout(p=0.2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1),
            torch.nn.Dropout(p=0.2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1),
            torch.nn.Dropout(p=0.2))
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 512, bias=True)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2))
        self.fc2 = torch.nn.Linear(512, 128, bias=True)
        self.fc3 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, x):
        x = torch.tensor(x[None,]).float()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ChessDataset(Dataset):
    def __init__(self, data_frame):
        self.fens = torch.from_numpy(np.array([*map(fen_to_bit_vector, data_frame["FEN"])], dtype=np.float32))
        self.evals = torch.Tensor([[x] for x in data_frame["Evaluation"]])
        self._len = len(self.evals)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.fens[index], self.evals[index]


def fen_to_bit_vector(fen):
    # piece placement - lowercase for black pieces, uppercase for white pieces. numbers represent consequtive spaces. / represents a new row
    # active color - whose turn it is, either 'w' or 'b'
    # castling rights - which castling moves are still legal K or k for kingside and Q or q for queenside, '-' if no legal castling moves for either player
    # en passant - if the last move was a pawn moving up two squares, this is the space behind the square for the purposes of en passant
    # halfmove clock - number of moves without a pawn move or piece capture, after 50 of which the game is a draw
    # fullmove number - number of full turns starting at 1, increments after black's move

    # Example FEN of starting position
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

    parts = re.split(" ", fen)
    piece_placement = re.split("/", parts[0])
    active_color = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_clock = int(parts[5])

    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)

    # piece to layer structure taken from reference [1]
    piece_to_layer = {
        'R': 1,
        'N': 2,
        'B': 3,
        'Q': 4,
        'K': 5,
        'P': 6,
        'p': 7,
        'k': 8,
        'q': 9,
        'b': 10,
        'n': 11,
        'r': 12
    }

    castling = {
        'K': (7, 7),
        'Q': (7, 0),
        'k': (0, 7),
        'q': (0, 0),
    }

    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)

    if en_passant != '-':
        bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1

    if castling_rights != '-':
        for char in castling_rights:
            bit_vector[0, castling[char][0], castling[char][1]] = 1

    if active_color == 'w':
        bit_vector[0, 7, 4] = 1
    else:
        bit_vector[0, 0, 4] = 1

    if halfmove_clock > 0:
        c = 7
        while halfmove_clock > 0:
            bit_vector[0, 3, c] = halfmove_clock % 2
            halfmove_clock = halfmove_clock // 2
            c -= 1
            if c < 0:
                break

    if fullmove_clock > 0:
        c = 7
        while fullmove_clock > 0:
            bit_vector[0, 4, c] = fullmove_clock % 2
            fullmove_clock = fullmove_clock // 2
            c -= 1
            if c < 0:
                break

    return bit_vector


def eval_to_int(evaluation):
    try:
        res = int(evaluation)
    except ValueError:
        res = 10000 if evaluation[1] == '+' else -10000
    return res / 100


def AdamW_main():
    MAX_DATA = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    print("Preparing Training Data...")
    train_data = pd.read_csv("chessData.csv")
    train_data = train_data[:MAX_DATA]
    train_data["Evaluation"] = train_data["Evaluation"].map(eval_to_int)
    trainset = ChessDataset(train_data)

    print("Preparing Test Data...")
    test_data = pd.read_csv("tactic_evals.csv")
    test_data = test_data[:MAX_DATA]
    test_data["Evaluation"] = test_data["Evaluation"].map(eval_to_int)
    testset = ChessDataset(test_data)

    batch_size = 100

    print("Converting to pytorch Dataset...")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters())

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # denominator for loss should represent the number of positions evaluated
                # independent of the batch size
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (2000 * len(labels))))
                running_loss = 0.0

    print('Finished Training')

    PATH = 'chess.pth'
    torch.save(net.state_dict(), PATH)

    print('Evaluating model')

    count = 0
    total_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print("Correct eval: {}, Predicted eval: {}, loss: {}".format(labels, outputs, loss))

            # count should represent the number of positions evaluated
            # independent of the batch size
            count += len(labels)
            total_loss += loss
            if count % 10000 == 0:
                print('Average error of the model on the {} tactics positions is {}'.format(count, loss / count))
    # print('Average error of the model on the {} tactics positions is {}'.format(count, loss/count))
