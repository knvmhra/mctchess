A bot that plays chess using a neural evaluation and MCTS, based on AlphaGo.

### Playing the bot:

Clone the repo, navigate to the root, and run:
```
mkdir yourvenv
python -m venv yourvenv
source yourvenv/bin/activate
pip install torch numpy chess tqdm
```

Edit play.py with the device you'd like to use for inference, a FEN if you want to set up a position, and a checkpoint (one is available on github, or you can train your own):
```
if __name__ == "__main__":
	device = torch.device("mps") # or "cpu" or "cuda"
	game = BotVsHuman(
	None, #None is the default start position
	"checkpoints/V1_checkpoint_epoch_6.pth",
	device
	)
	game.play()
```

Make your moves in SAN when playing the bot. The bot responds with UCI moves and its evaluation of the position. The bot always goes first. 

### Training the bot:

**Gather training data**

Collect training games into a PGN file. Create a directory at the root of your project called datasets.

Provide a path to your PGN file and a name for the file containing the generated dataset in serialise.py:
```
if __name__ == "__main__":
	X, P, V = generate_data(
	"path_to_games.pgn", 
	15000 #maximum number of games to parse.
	)
	np.savez("datasets/name_of_data", X, P, V)
```
Run serialise.py to create a .npz containing the encoded board position, moves, and evaluations

**Configure training loop**

Create a directory at the root of your project called checkpoints.  

In train.py, choose a training device and provide a path to your training data. You can also choose the batch size (lower sizes will take longer to train)
```
#setup

device = torch.device("mps") #or "cpu" or "cuda"

dataset = TrainingData('datasets/your_generated_data.npz')

dataloader = DataLoader(dataset, batch_size= 128, shuffle=True)

```

Next, provide a name for your model (avoid other changes to the name and do not use underscores, the training script parses names to resume training from the latest checkpoint in the checkpoints directory):
```

checkpoint_path = os.path.join(checkpoint_dir, f'yourmodelname_checkpoint_epoch_{epoch+1}.pth')
torch.save(...
...

```

Now, just run train.py. You should see something like this:

![[Screenshot 2024-09-17 at 6.36.08 PM.png]]

That's it! Once you're done training, just follow the steps in the "Playing the bot" section to play against your bot. 