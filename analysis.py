import pandas as pd
import os
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
import torch
import numpy as np

### ### ### Settings ### ### ###
### Causal LLMs
c_models = ['BSC-LT/salamandra-7b', 'facebook/xglm-7.5B', 'catallama/CataLlama-v0.1-Base', 'catallama/CataLlama-v0.2-Base', 'gplsi/Aitana-6.3B', 'BSC-LT/ALIA-40b', 'bigscience/bloom-7b1']

### MLM LLMs
mlm_models = ['BSC-LT/RoBERTa-ca']

### Input data
df = pd.read_csv("NormaUsInterferencia/datasets/dataset.csv")
df['s1norm'] = df.apply(lambda row: row['Masked'].replace('<mask>', row['Resposta normativa']), axis=1)
df['s2nonorm'] = df.apply(lambda row: row['Masked'].replace('<mask>', row['Resposta no normativa freqüent']), axis=1)

### Output data
out_path = 'NormaUsInterferencia/resultats/resultats.csv'
### ### ### ### ###


###################################
### ### ### Causal LLMs ### ### ###
###################################

def calculate_sequence_nll(model, tokenizer, sequence: str) -> float:
    """
    Calculates the Negative Log-Likelihood (NLL) of a given sequence using a causal language model.
    A lower NLL indicates a higher probability.

    Args:
        model: A pre-trained causal language model (e.g., GPT-2).
        tokenizer: The tokenizer corresponding to the model.
        sequence: The input text sequence.

    Returns:
        The negative log-likelihood (float) of the sequence.
        Returns float('inf') if an error occurs or the sequence cannot be processed.
    """
    try:
        # Tokenize the input sequence
        # add_special_tokens=True adds tokens like <bos> (beginning of sequence) if the model uses them.
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)

        # Move inputs to the same device as the model (e.g., 'cuda' if available)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Get model outputs (logits)
        # labels=inputs.input_ids tells the model to calculate the loss against the input_ids themselves
        # This is how we get the per-token negative log-likelihoods
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(**inputs, labels=inputs['input_ids'])

        # The loss returned by AutoModelForCausalLM is the average negative log-likelihood
        # across all tokens (excluding padding, and potentially the first token depending on implementation).
        # To get the total NLL for the sequence, we need to consider the number of tokens.

        # However, a more direct way for sequence probability is to look at the per-token log-likelihoods.
        # The 'loss' from model(**inputs, labels=inputs.input_ids) is often already the average NLL,
        # which is exactly what we want for comparison. Let's verify this.

        # For comparison, the *average* NLL is generally sufficient and more robust than total NLL
        # if comparing sequences of different lengths.
        # But for minimal pairs of same length, total NLL is also fine.

        # The 'loss' attribute directly gives the negative log-likelihood (NLL) averaged over the sequence.
        # This is the standard way to get perplexity (exp(loss)) or sequence likelihood from LM's.
        # A smaller loss means higher likelihood.
        nll = outputs.loss.item()
        return nll

    except Exception as e:
        print(f"Error calculating NLL for sequence '{sequence}': {e}")
        return float('inf') # Return infinity for errors


def compare_sequence_likelihoods(model, tokenizer, sequence1: str, sequence2: str):
    """
    Compares the likelihood of two sequences using their Negative Log-Likelihood (NLL).

    Args:
        model: A pre-trained causal language model.
        tokenizer: The tokenizer corresponding to the model.
        sequence1: The first sequence to compare.
        sequence2: The second sequence to compare.
    """
    print(f"Comparing likelihoods using model: {model.config._name_or_path}")
    nll1 = calculate_sequence_nll(model, tokenizer, sequence1)
    nll2 = calculate_sequence_nll(model, tokenizer, sequence2)

    print(f"\nSequence 1: '{sequence1}'")
    print(f"  Negative Log-Likelihood (NLL): {nll1:.4f}")

    print(f"Sequence 2: '{sequence2}'")
    print(f"  Negative Log-Likelihood (NLL): {nll2:.4f}")

    if nll1 < nll2:
        print(f"\nConclusion: Sequence 1 is more likely than Sequence 2 (Lower NLL).")
    elif nll2 < nll1:
        print(f"\nConclusion: Sequence 2 is more likely than Sequence 1 (Lower NLL).")
    else:
        print(f"\nConclusion: Both sequences have similar likelihoods.")
    return([nll1,nll2])




# Check if there's already a file with some results saved
# If so, load it
if os.path.isfile(out_path):
   df_out = pd.read_csv(out_path)
   already_processed = df_out['model'].unique()
   print(f'Loading previous results. Models already processed {already_processed}')
# If not, start from scratch
else:
  print('No previous results found. Starting from scratch')
  df_out = None
  already_processed = []

for model_name in c_models:
  # Check if the model was already processed (since this is expensive to re-compute)
  # If it wasn't processed, go line by line computing NLLs for the two sentences and save results
  if model_name not in already_processed:
    print(f'...processing {model_name}')
    #Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # Get a fresh list to save results per row
    outs = []
    # Iterate over rows, getting NLL for each of the two sentences we're comparing
    for index, row in df.iterrows():
      print(f'... ... current index: {index}')
      p1, p2 = compare_sequence_likelihoods(model, tokenizer, row['s1norm'], row['s2nonorm'])
      new_row = {
        'index': index,
        'model': model_name,
        'tipus': row["Tipus d'error"], 
        'snormatiu':  row['s1norm'],
        'sno_normatiu': row['s2nonorm'],
        'pnormatiu': p1,
        'pno_normatiu': p2
      }
      outs.append(new_row)
  # If there wasn't data storing previous results, this is the first batch
    if df_out is None:
      df_out = pd.DataFrame(outs)
  # Otherwise concatenate new data to old data 
    else:
      df_out = pd.concat([df_out, pd.DataFrame(outs)], ignore_index=True)
  df_out.to_csv(out_path, index=False)


##################################

###################################
### ### ### Masked LLMs ### ### ###
###################################



def get_nll_mlm(model, tokenizer, sequence1: str, sequence2: str):
    """
    Calculates the negative log-likelihood (NLL) of two sentences using a
    masked language model. 

    The function computes the pseudo-NLL by iterating through each token in
    the sentence, masking it, and then calculating the log probability of the
    original token at that masked position.

    Args:
        model: A Hugging Face masked language model.
        tokenizer: The tokenizer corresponding to the model.
        sequence1 (str): The first sentence.
        sequence2 (str): The second sentence.

    Returns:
        A tuple containing the negative log-likelihoods for sequence1 and
        sequence2, respectively.
    """
    def calculate_sentence_nll(sequence) -> float:
        """Helper function to calculate NLL for a single sentence."""
        # Tokenize the input sequence, adding special tokens for the model
        # and returning the output as PyTorch tensors.
        tokenized_input = tokenizer(sequence, return_tensors="pt")
        input_ids = tokenized_input["input_ids"]
        
        # We don't want to mask the special tokens [CLS] and [SEP],
        # so we'll iterate over the tokens in between.
        # The first token is [CLS] and the last is [SEP].
        start_token_idx = 1
        end_token_idx = len(input_ids[0]) - 2

        total_nll = 0.0

        # Iterate through each token in the sequence to mask it one by one
        for i in range(start_token_idx, end_token_idx + 1):
            # Create a copy of the input_ids to modify
            masked_input_ids = input_ids.clone()
            
            # Store the original token ID that we are masking
            original_token_id = masked_input_ids[0, i].item()
            
            # Replace the token at the current position with the mask token
            masked_input_ids[0, i] = tokenizer.mask_token_id

            # Get model predictions without calculating gradients
            with torch.no_grad():
                outputs = model(masked_input_ids)
                logits = outputs.logits

            # Get the logits for the masked token
            masked_token_logits = logits[0, i, :]
            
            # Calculate the log probability of the original token
            # We use log_softmax to get log probabilities from logits
            log_probs = torch.nn.functional.log_softmax(masked_token_logits, dim=0)
            
            # Get the log probability of the original token
            token_log_prob = log_probs[original_token_id].item()
            
            # Add the negative log probability to the total
            total_nll -= token_log_prob
            
        return total_nll

    print(f"Comparing likelihoods using model: {model.config._name_or_path}")
    nll1 = calculate_sentence_nll(sequence1)
    nll2 = calculate_sentence_nll(sequence2)

    print(f"\nSequence 1: '{sequence1}'")
    print(f"  Negative Log-Likelihood (NLL): {nll1:.4f}")

    print(f"Sequence 2: '{sequence2}'")
    print(f"  Negative Log-Likelihood (NLL): {nll2:.4f}")

    if nll1 < nll2:
        print(f"\nConclusion: Sequence 1 is more likely than Sequence 2 (Lower NLL).")
    elif nll2 < nll1:
        print(f"\nConclusion: Sequence 2 is more likely than Sequence 1 (Lower NLL).")
    else:
        print(f"\nConclusion: Both sequences have similar likelihoods.")

    return nll1, nll2


if os.path.isfile(out_path):
   df_out = pd.read_csv(out_path)
   already_processed = df_out['model'].unique()
   print(f'Loading previous results. Models already processed {already_processed}')
# If not, start from scratch
else:
  print('No previous results found. Starting from scratch')
  df_out = None
  already_processed = []

for model_name in mlm_models:
  # Check if the model was already processed (since this is expensive to re-compute)
  # If it wasn't processed, go line by line computing NLLs for the two sentences and save results
  if model_name not in already_processed:
    print(f'...processing {model_name}')
    #Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == 'BSC-LT/RoBERTa-ca': #set masked token manually
      tokenizer.mask_token = '<mask>'
    model = AutoModelForMaskedLM.from_pretrained(model_name, device_map = 'auto')
    # Get a fresh list to save results per row
    outs = []
    # Iterate over rows, getting NLL for each of the two sentences we're comparing
    for index, row in df.iterrows():
      print(f'... ... current index: {index}')
      p1, p2 = get_nll_mlm(model, tokenizer, row['s1norm'], row['s2nonorm'])
      new_row = {
        'index': index,
        'model': model_name,
        'tipus': row["Tipus d'error"], 
        'snormatiu':  row['s1norm'],
        'sno_normatiu': row['s2nonorm'],
        'pnormatiu': p1,
        'pno_normatiu': p2
      }
      outs.append(new_row)
  # If there wasn't data storing previous results, this is the first batch
    if df_out is None:
      df_out = pd.DataFrame(outs)
  # Otherwise concatenate new data to old data 
    else:
      df_out = pd.concat([df_out, pd.DataFrame(outs)], ignore_index=True)
  df_out.to_csv(out_path, index=False)