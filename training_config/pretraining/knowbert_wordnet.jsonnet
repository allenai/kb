{
    "vocabulary": {
        "directory_path": "s3://allennlp/knowbert/models/vocabulary_wordnet.tar.gz",
    },

    "dataset_reader": {
        "type": "multitask_reader",
        "datasets_for_vocab_creation": [],
        "dataset_readers": {
            "language_modeling": {
                "type": "multiprocess",
                "base_reader": {
                    "type": "bert_pre_training",
                    "tokenizer_and_candidate_generator": {
                        "type": "bert_tokenizer_and_candidate_generator",
                        "entity_candidate_generators": {
                            "wordnet": {"type": "wordnet_mention_generator",
                                        "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"},
                        },
                        "entity_indexers":  {
                            "wordnet": {
                                   "type": "characters_tokenizer",
                                   "tokenizer": {
                                       "type": "word",
                                       "word_splitter": {"type": "just_spaces"},
                                   },
                                   "namespace": "entity"
                                }
                        },
                        "bert_model_type": "bert-base-uncased",
                        "do_lower_case": true,
                    },
                    "lazy": true,
                    "mask_candidate_strategy": "full_mask",
                },
                "num_workers": 1,
            },
            "entity_linking": {
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
                "should_remap_span_indices": false,
            },
        }
    },

    "iterator": {
        "type": "multitask_iterator",
        "names_to_index": ["language_modeling", "entity_linking"],
        "iterate_forever": true,

        "sampling_rates": [0.8, 0.2],

        "iterators": {
            "language_modeling": {
                "type": "multiprocess",
                "base_iterator": {
                    "type": "self_attn_bucket",
                    "batch_size_schedule": "base-24gb-fp32",
                    "iterator": {
                        "type": "bucket",
                        "batch_size": 32,
                        "sorting_keys": [["tokens", "num_tokens"]],
                        "max_instances_in_memory": 2500,
                    }
                },
                "num_workers": 1,
            },
            "entity_linking": {
                "type": "self_attn_bucket",
                "batch_size_schedule": "base-24gb-fp32",
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
                    "iterate_forever": true,
                    "id_type": "wordnet",
                    "use_nsp_label": true,
                }
            },
        },
    },

    "train_data_path": {
        "language_modeling": "/home/matthewp/data/wikipedia_torontobooks_for_bert/*.txt",
        "entity_linking": "s3://allennlp/knowbert/wordnet/semcor_and_wordnet_examples.json",
    },

    "model": {
        "type": "knowbert",
        "bert_model_name": "bert-base-uncased",
        "model_archive": "/home/matthewp/data/knowbert/models/wordnet_linker.tar.gz",
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
            "lr": 1e-4,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,

            "parameter_groups": [
                // all layer norm and bias in bert have no weight decay and small lr
                [["pretrained_bert.*embeddings.*LayerNorm",
                  "pretrained_bert.*encoder.layer.[0-9]\\..*LayerNorm", "pretrained_bert.*encoder.layer.[0-9]\\..*bias",
                  "pretrained_bert.*cls.*LayerNorm", "pretrained_bert.*cls.*bias",
                  "pretrained_bert.*pooler.*bias"], {"lr": 2e-5, "weight_decay": 0.0}],
                // remaining parameters have small lr
                [["pretrained_bert.*embeddings[^L]+$", "pretrained_bert.*pooler.*weight", "pretrained_bert.*cls[^L]+weight",
                  "pretrained_bert.*encoder.layer.[0-9]\\.[^L]+weight"], {"lr": 2e-5, "weight_decay": 0.01}],
                [[
                  "pretrained_bert.*encoder.layer.1[0-1].*LayerNorm", "pretrained_bert.*encoder.layer.1[0-1].*bias"],
                  {"lr": 5e-5, "weight_decay": 0.0}],
                [[
                  "pretrained_bert.*encoder.layer.1[0-1][^L]+weight"],
                  {"lr": 5e-5, "weight_decay": 0.01}],
                // other bias and layer norm have no weight decay
                [["soldered_kg.*LayerNorm", "soldered_kg.*layer_norm", "soldered_kg.*bias"],
                  {"weight_decay": 0.0}],
            ],
        },
        "gradient_accumulation_batch_size": 32,
        "num_epochs": 1,
        "num_steps_reset_metrics": 5000,

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 1,
            "num_steps_per_epoch": 240000,
            "cut_frac": 0.025,
        },
        "num_serialized_models_to_keep": 2,
        "model_save_interval": 600,
        "should_log_learning_rate": true,
        "cuda_device": 0,
    }

}
