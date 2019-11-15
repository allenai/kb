{
    "dataset_reader": {
        "type": "semeval2010_task8",
        "entity_masking": "entity_markers",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_and_candidate_generator",
            "bert_model_type": "s3://allennlp/knowbert/models/bert-base-uncased-tacred-entity-markers-vocab.txt",
            "do_lower_case": true,
            "entity_candidate_generators": {
                "wiki": {
                    "type": "wiki",
                },
                "wordnet": {
                    "type": "wordnet_mention_generator",
                    "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
                }
            },
            "entity_indexers": {
                "wiki": {
                    "type": "characters_tokenizer",
                    "namespace": "entity_wiki",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {
                            "type": "just_spaces"
                        }
                    }
                },
                "wordnet": {
                    "type": "characters_tokenizer",
                    "namespace": "entity_wordnet",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {
                            "type": "just_spaces"
                        }
                    }
                }
            }
        }
    },
    "iterator": {
        "iterator": {
            "type": "basic",
            "batch_size": 32
        },
        "type": "self_attn_bucket",
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "model": {
            "type": "from_archive",
            "archive_file": "s3://allennlp/knowbert/models/knowbert_wiki_wordnet_model.tar.gz",
        },
        "type": "simple-classifier",
        "bert_dim": 768,
        "concat_word_a_b": true,
        "include_cls": false,
        "metric_a": {
            "type": "semeval2010_task8_metric"
        },
        "num_labels": 19,
        "task": "classification"
    },
    "train_data_path": "/home/matthewp/data/semeval2010_task8/train.json",
    "validation_data_path": "/home/matthewp/data/semeval2010_task8/dev.json",
    "trainer": {
        "cuda_device": 0,
        "gradient_accumulation_batch_size": 32,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 3,
            "num_steps_per_epoch": 234.375
        },
        "num_epochs": 3,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "b2": 0.98,
            "lr": 5e-05,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+f1"
    },
    "vocabulary": {
        "directory_path": "s3://allennlp/knowbert/models/vocabulary_wordnet_wiki.tar.gz"
    }
}
