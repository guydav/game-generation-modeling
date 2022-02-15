import torch
import torch.nn.functional as f
from tqdm import tqdm


class LMSampler(object):
    """
    Given a tokenizer/decoder and a language model, performs simple sampling.
    To do: other generation strategies like beam search
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _top_masked(self, x, k=None):
        """
        Masks everything but the k top entries of x = [batch, values] as
        -infinity (1e10). Used to mask logits such that e^-infinity -> 0
        won't contribute to the sum of the denominator.
        """
        if k is None:
            return x
        else:
            values = torch.topk(x, k)[0]
            batch_mins = values[:, -1].view(-1, 1).expand_as(x)
            return torch.where(x < batch_mins,
                               torch.ones_like(x) * -1e10,
                               x)

    def generate(self, condition_text="", topk_sample=40, temperature=1,
                 length=512):
        """
        Given a condition text, generates a continuation.
        Args:
            condition_text: Text to condition on
            topk_sample: Possible choices are
                - An integer k to sample from the top k logits. 
                - k=1 implies no sampling, just chosing the top logit
                - k=None samples from the entire softmax output in a multinomial
                  fashion
            temperature: Softmax temperature (used to pre-scale the logits) to
                make softmax less peak-y
        """
        batch_size = 1  # fixed

        token_ids = self.tokenizer.encode(condition_text,
                                          add_special_tokens=False)

        self.model.eval()

        length = length-len(token_ids)

        prev = torch.tensor(token_ids, device=self.device,
                            dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        output = prev

        with torch.no_grad():
            for i in tqdm(range(length), desc="Generating continuation", leave=False):
                # TODO: Speed up GPT-1 with past, precomputed hidden states as well
                logits = self.model(output).logits
                if isinstance(logits, tuple):
                    logits = logits[0]

                logits = logits[:, -1, :] / temperature
                logits = self._top_masked(logits, k=topk_sample)
                log_probs = f.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)

                # Concatenate
                output = torch.cat((output, prev), dim=1)

        output_token_ids = output.cpu().numpy().tolist()[0]
        result = self.tokenizer.decode(output_token_ids)

        return result