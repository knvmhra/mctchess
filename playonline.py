import pygame
from PIL import Image, ImageDraw
import chess.svg
from mcts import MCTS, Node
from constants import POLICY_MAP
from chess import Board, Move
from model import NetV1
import torch
import io

class BotVsHuman:
    def __init__(self, start_pos, model_path: str, device):
        if start_pos is None:
            self.board = Board()
        else:
            self.board = Board(start_pos)
        self.device = device
        self.model = self._load_model(model_path)
        self.mcts = MCTS(self.model, self.device, num_simulations=1000)

        # Initialize pygame window
        self.window_size = (512, 512)  # window size for pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Chess Game')

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = NetV1(cfg=checkpoint['model_config'])
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def draw_board(self):
        """Convert the chess board to an image using python-chess and Pillow, then render it with pygame."""
        # Generate SVG from the board
        svg_board = chess.svg.board(self.board)

        # Convert SVG to an image using Pillow
        image = Image.open(io.BytesIO(svg_board.encode('utf-8')))

        # Resize image to fit window
        image = image.resize(self.window_size)

        # Convert the Pillow image to a format Pygame can use
        mode = image.mode
        size = image.size
        data = image.tobytes()

        surface = pygame.image.fromstring(data, size, mode)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def get_player_move(self):
        """Handle player move input via pygame."""
        move_str = ''
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        try:
                            move = self.board.parse_san(move_str)
                            return move
                        except:
                            print("Invalid move, try again")
                            move_str = ''
                    else:
                        move_str += event.unicode

    def get_bot_move(self, root: Node):
        simulated_root = self.mcts.search(root)
        move = simulated_root.optimal_policy()
        return move, simulated_root

    def play(self):
        root = Node(board=self.board.copy())
        while True:
            self.draw_board()  # Draw the current board state
            
            # Bot move
            bot_move_idx, root = self.get_bot_move(root)
            self.board.push(Move.from_uci(POLICY_MAP[bot_move_idx]))
            print(f"Bot played: {POLICY_MAP[bot_move_idx]}, eval: {root.value()}")
            self.draw_board()  # Redraw the board with bot's move

            # Player move
            player_move = self.get_player_move()
            self.board.push(player_move)
            print(self.board)
            player_move_idx = POLICY_MAP.index(player_move.__str__())
            root = root.make_node_move(bot_move_idx, player_move_idx)
            self.draw_board()  # Redraw after player move

if __name__ == "__main__":
    device = torch.device("mps")
    game = BotVsHuman(
        None,
        "/Users/kmwork/mctchess/checkpoints/V1_checkpoint_epoch_6.pth",
        device
    )
    game.play()