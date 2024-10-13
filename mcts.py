from chess import Board, WHITE, BLACK, Move
import numpy as np
from constants import POLICY_MAP
from serialise import serialise_board
import torch
from typing import Dict, Tuple
from typing_extensions import Self

class Node:
    def __init__(self, board: Board, prior: float= 0, visit_count: int = 0, value_sum: float = 0):
        self.visit_count = visit_count
        self.value_sum = value_sum
        self.state = board
        self.prior = prior
        self.children: Dict[int, Node] = {} #k = policy index, v = Node

    def tree_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.tree_depth() for child in self.children.values())
    
    def child_counts(self):
        for k in self.children:
            print(POLICY_MAP[k])
            print(self.children[k].visit_count)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        else:
            return self.value_sum / self.visit_count

    def expand(self, policy_vec: torch.Tensor, k: int = 0,  visit_count: int = 0, value_sum: float = 0) -> None:
        for i, p in enumerate(policy_vec.squeeze().cpu().numpy()):
            if p > 0 and (self.children.get(i, None) is None): #don't overwrite already expanded children
                prior = p.item()
                board = self.state.copy()
                board.push_uci(POLICY_MAP[i])
                self.children[i] = Node(board= board, prior= prior, visit_count= visit_count, value_sum= value_sum)

    def puct_score(self, child: Self, c=2) -> float:
        Q = child.value()
        U = child.prior * np.sqrt(self.visit_count) / (child.visit_count + 1)
        return Q + c * U

    def select_child(self) -> Tuple[int, Self]:
        best_score = -np.inf
        best_policy: int = -1
        best_child = None
        for policy, child in self.children.items():
            score = self.puct_score(child)
            if score > best_score:
                best_score = score
                best_policy = policy
                best_child = child
        return best_policy, best_child

    def optimal_policy(self) -> int:
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        policies = [p for p in self.children.keys()]
        return policies[np.argmax(visit_counts).item()]

    def make_node_move(self, my_policy: int, opp_policy: int) -> Self:
        #it is the bots turn to play
        #traverse the tree to arrive at the new root by applying my (the bot's) policy and the opponent policy
 
        if opp_policy in self.children[my_policy].children: #have i simulated your move? if so traverse the tree
            new_root = self.children[my_policy].children[opp_policy]
            del self.children[my_policy].children #helping the GC
            del self.children
            return new_root
        else: #opponent move was unexpected (should not happen often)
            new_root_state = self.state.copy()
            new_root_state.push(Move.from_uci(POLICY_MAP[my_policy]))
            new_root_state.push(Move.from_uci(POLICY_MAP[opp_policy]))
            out = Node(new_root_state)
            del self.children
            return out
        

class MCTS:
    def __init__(self, model: torch.nn.Module, device: torch.device, num_simulations: int = 100):
        self.sims = num_simulations
        self.model = model
        self.device = device

    def query_model(self, state: Board) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            board = torch.tensor(serialise_board(state), dtype= torch.float32).to(self.device)
            policy, value = self.model(board.unsqueeze(0)) #unsqueeze to add batch dimension

        policy = torch.nn.Softmax(dim= -1)(policy)
        legal_mask = torch.tensor([1 if Move.from_uci(mv) in state.legal_moves else 0 for mv in POLICY_MAP]).to(self.device)
        policy = (policy.squeeze() * legal_mask)

        return policy, value.item()

    def search(self, root: Node, branching_factor: int= 0):
        self.model.eval()

        for _ in range(self.sims):

            node = root
            search_path: list[Node] = [node]

            while node.is_expanded():
                _, node = node.select_child()
                search_path.append(node) #SELECTION

            if node.state.is_game_over():
                outcome = node.state.outcome()
                if outcome.winner == WHITE:
                    value = 1
                elif outcome.winner == BLACK:
                    value = - 1
                else:
                    value = 0
            else:
                policy, value = self.query_model(node.state) #SIMULATION
                node.expand(policy, branching_factor) #EXPANSION
            self._backpropagate(search_path, value, node.state.turn) #BACKPROPAGATION

        return root

    @staticmethod
    def _backpropagate(search_path: list[Node], value: float, turn: bool):
        for n in search_path:
            n.value_sum += value if n.state.turn == turn else -value
            n.visit_count += 1
