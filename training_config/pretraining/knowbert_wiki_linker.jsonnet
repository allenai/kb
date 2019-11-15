{
    "vocabulary": {
        "directory_path": "s3://allennlp/knowbert/models/vocabulary_wiki.tar.gz"
    },

    "dataset_reader": {
        "type": "aida_wiki_linking",
        "entity_disambiguation_only": false,
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
           "namespace": "entity",
        },
        "should_remap_span_indices": false,
    },

    "iterator": {
        "type": "self_attn_bucket",
        "batch_size_schedule": "base-12gb-fp32",
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
            "id_type": "wiki",
            "use_nsp_label": false,
        },
    },

    "train_data_path": "s3://allennlp/knowbert/wiki_entity_linking/aida_train.txt",
    "validation_data_path": "s3://allennlp/knowbert/wiki_entity_linking/aida_dev.txt",

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "mode": "entity_linking",
        "soldered_layers": {"wiki": 9},
        "soldered_kgs": {
            "wiki": {
                "type": "soldered_kg",
                "entity_linker": {
                    "type": "entity_linking_with_candidate_mentions",
                    "entity_embedding": {
                        "vocab_namespace": "entity",
                        "embedding_dim": 300,
                        "pretrained_file": "s3://allennlp/knowbert/wiki_entity_linking/entities_glove_format.gz",
                        "trainable": false,
                        "sparse": false
                    },
                    "contextual_embedding_dim": 768,
                    "span_encoder_config": {
                        "hidden_size": 300,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "intermediate_size": 1024
                    },
                },
                "span_attention_config": {
                    "hidden_size": 300,
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
        "gradient_accumulation_batch_size": 32,
        "num_epochs": 10,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 10,
            "num_steps_per_epoch": 434,
        },
        "num_serialized_models_to_keep": 2,
        "should_log_learning_rate": true,
        "cuda_device": 0,
        "validation_metric": "+wiki_el_f1",
    }

}
