from mcts import MCTS, Node
from constants import POLICY_MAP
from chess import Board, Move
from model import NetV1
import torch

class BotVsHuman:
    def __init__(self, start_pos, model_path: str, device):
        if start_pos == None:
            self.board = Board()
        else:
            self.board = Board(start_pos)
        self.device = device
        self.model = self._load_model(model_path)
        self.mcts = MCTS(self.model, self.device, num_simulations=1000)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = NetV1()
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_bot_move(self, root: Node):
        simulated_root = self.mcts.search(root)
        move = simulated_root.optimal_policy()
        return move, simulated_root
    
    def get_player_move(self):
        player_input = input()
        try:
            move = self.board.parse_san(player_input)
        except:
            print("invalid move, try again")
            move = self.get_player_move()
        return move
        
    def play(self):
        root = Node(board= self.board.copy())
        while True:
            bot_move_idx, root = self.get_bot_move(root)
            self.board.push(Move.from_uci(POLICY_MAP[bot_move_idx]))
            print(self.board)
            print(f"bot played: {POLICY_MAP[bot_move_idx]}, eval: {root.value()}")
            player_move = self.get_player_move()
            self.board.push(player_move)
            print(self.board)
            player_move_idx = POLICY_MAP.index(player_move.__str__())
            root = root.make_node_move(bot_move_idx, player_move_idx)

if __name__ == "__main__":
    device = torch.device("mps")
    game = BotVsHuman(
        None,
        "/Users/kmwork/train_ochess/checkpoints/V1mini_checkpoint_epoch_20.pth",
        device
    )
    game.play()