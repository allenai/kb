{
    "vocabulary": {
        "directory_path": "s3://allennlp/knowbert/models/vocabulary_wordnet.tar.gz",
    },

    "dataset_reader": {
        "type": "wordnet_fine_grained",
        "wordnet_entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl",
        "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": true,
                "max_pieces": 512,
            },
        },
        "entity_indexer": {
           "type": "characters_tokenizer",
           "tokenizer": {
               "type": "word",
               "word_splitter": {"type": "just_spaces"},
           },
           "namespace": "entity"
        },
        "is_training": true,
        "should_remap_span_indices": false
    },

    "train_data_path": "s3://allennlp/knowbert/wordnet/semcor_and_wordnet_examples.json",

    "iterator": {
        "type": "cross_sentence_linking",
        "batch_size": 32,
        "entity_indexer": {
            "type": "characters_tokenizer",
            "tokenizer": {
                "type": "word",
                "word_splitter": {"type": "just_spaces"},
            },
            "namespace": "entity"
        },
        "bert_model_type": "bert-base-uncased",
        "do_lower_case": true,
        // this is ignored
        "mask_candidate_strategy": "none",
        "max_predictions_per_seq": 0,
        "id_type": "wordnet",
        "use_nsp_label": false,
    },

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "mode": "entity_linking",
        "soldered_layers": {"wordnet": 9},
        "soldered_kgs": {
            "wordnet": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "loss_type": "softmax",
                    "concat_entity_embedder": {
                        "type": "wordnet_all_embeddings",
                        "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl",
                        "embedding_file": "s3://allennlp/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5",
                        "vocab_file": "s3://allennlp/knowbert/wordnet/wordnet_synsets_mask_null_vocab.txt",
                        "entity_dim": 200,
                        "entity_h5_key": "tucker_gensen",
                    },
                    "contextual_embedding_dim": 768,
                    "span_encoder_config": {
                        "hidden_size": 200,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "intermediate_size": 1024
                    },
                },
                "span_attention_config": {
                    "hidden_size": 200,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "intermediate_size": 1024
                },
            },
        },
    },

    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-3,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },
        "num_epochs": 5,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            // semcor + examples batch size=32
            "num_steps_per_epoch": 2470,
        },
        "num_serialized_models_to_keep": 1,
        "should_log_learning_rate": true,
        "cuda_device": 0,
    }

}
