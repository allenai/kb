
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params

import torch

if __name__ == '__main__':
    archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
    params = Params({"archive_file": archive_file})

    # load model and batcher
    model = ModelArchiveFromParams.from_params(params=params)
    batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')

    sentences = ["Paris is located in [MASK].", "La Mauricie National Park is located in [MASK]."]

    mask_id = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab['[MASK]']
    for batch in batcher.iter_batches(sentences):
        model_output = model(**batch)
        token_mask = batch['tokens']['tokens'] == mask_id

        # (batch_size, timesteps, vocab_size)
        prediction_scores, _ = model.pretraining_heads(
                model_output['contextual_embeddings'], model_output['pooled_output']
        )

        mask_token_probabilities = prediction_scores.masked_select(token_mask.unsqueeze(-1)).view(-1, prediction_scores.shape[-1])  # (num_masked_tokens, vocab_size)

        predicted_token_ids = mask_token_probabilities.argmax(dim=-1)

        predicted_tokens = [batcher.tokenizer_and_candidate_generator.bert_tokenizer.ids_to_tokens[int(i)]
            for i in predicted_token_ids]

        print(predicted_tokens)

