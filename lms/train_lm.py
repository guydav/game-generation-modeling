import argparse
from lm_datasets import GameDescriptionGPT2Dataset
from language_model import GPT2LanguageModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup_proportion', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    dataset = GameDescriptionGPT2Dataset(chunk_size=args.chunk_size)
    model = GPT2LanguageModel()

    model.fit(dataset=dataset,
              batch_size=args.batch_size,
              num_epochs=args.epochs,
              learning_rate=args.learning_rate,
              weight_decay=args.weight_decay,
              warmup_proportion=args.warmup_proportion)