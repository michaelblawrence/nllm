pub mod transformer {
    use std::{
        collections::{HashMap, HashSet, VecDeque},
        ops::Deref,
    };

    use anyhow::{Context, Result};
    use itertools::Itertools;
    use serde::{Deserialize, Serialize};
    use tracing::instrument;

    use crate::ml::{
        embeddings::TrainBatchConfig,
        transformer::{
            decoder::{builder::DecoderBuilder, Decoder},
            linear::Linear,
            solver::source::OptimizerSource,
        },
        LayerValues, NetworkActivationMode, NodeValue, RngStrategy, SamplingRng,
    };

    #[derive(Serialize, Deserialize)]
    pub struct CharacterTransformer {
        network: Decoder,
        vocab: HashMap<char, usize>,
        rng: RngStrategy,
        context_length: usize,
    }

    const CONTROL_VOCAB: char = '\x00';

    impl CharacterTransformer {
        pub fn from_builder(
            builder: DecoderBuilder,
            vocab: HashSet<char>,
            rng: RngStrategy,
        ) -> Result<Self> {
            Self::from_builder_ordered(builder, &vocab.into_iter().collect_vec(), rng)
        }

        pub fn from_builder_ordered(
            builder: DecoderBuilder,
            vocab: &[char],
            rng: RngStrategy,
        ) -> Result<Self> {
            let context_length = builder.sequence_len();
            let vocab = Self::from_oredered_vocab(vocab);
            let builder = builder.with_target_vocab_size(vocab.len());
            Ok(Self {
                network: builder.build()?,
                vocab,
                rng,
                context_length,
            })
        }

        fn from_oredered_vocab(vocab: &[char]) -> HashMap<char, usize> {
            [CONTROL_VOCAB]
                .into_iter()
                .chain(vocab.into_iter().copied())
                .enumerate()
                .map(|(i, word)| (word, i))
                .collect()
        }

        pub fn snapshot(&self) -> Result<String> {
            Ok(serde_json::to_string(self)?)
        }

        pub fn snapshot_pretty(&self) -> Result<String> {
            Ok(serde_json::to_string_pretty(self)?)
        }

        #[instrument(level = "info", name = "embed_train", skip_all)]
        pub fn train<B: Into<TrainBatchConfig>, T: Deref<Target = O>, O: OptimizerSource>(
            &mut self,
            corpus: &str,
            optimizer: T,
            batch_size: B,
        ) -> Result<NodeValue> {
            let batch_size: TrainBatchConfig = batch_size.into();
            let batches = self.into_batches(corpus, batch_size);
            let (encoder_output, encoder_mask) = (None, None);

            #[cfg(feature = "threadpool")]
            use rayon::prelude::*;

            let batches_iter = {
                #[cfg(feature = "threadpool")]
                {
                    batches.par_iter()
                }
                #[cfg(not(feature = "threadpool"))]
                {
                    batches.iter()
                }
            };
            let batches_forward_pass_costs = batches_iter
                .map(|(input_sequence, targets)| {
                    let decoder_outputs = self.network.forward_training(
                        &input_sequence,
                        encoder_output,
                        encoder_mask,
                    )?;

                    let output_gradients = decoder_outputs.rows_iter().zip_eq(targets).map(
                        |(decoder_output, target)| {
                            let vocab_len = self.vocab.len();
                            let expected_outputs = (0..vocab_len)
                                .map(move |i| if i == *target { 1.0 } else { 0.0 })
                                .collect();

                            let post_apply_mode = NetworkActivationMode::SoftMaxCrossEntropy;
                            let probabilities = post_apply_mode.apply(&decoder_output.into());

                            let train_error =
                                probabilities.cross_entropy_error(&expected_outputs)?;

                            let output_gradients =
                                probabilities.cross_entropy_error_d(&expected_outputs)?;

                            Ok((output_gradients, train_error.ave()))
                        },
                    );

                    let (output_gradients, train_error): (Vec<_>, Vec<_>) = output_gradients
                        .collect::<Result<Vec<(LayerValues, NodeValue)>>>()?
                        .into_iter()
                        .unzip();
                    let decoder_gradients = Linear::from_values(&output_gradients)?;
                    self.network.backward(
                        &input_sequence,
                        encoder_output,
                        encoder_mask,
                        decoder_gradients,
                    )?;

                    Ok(train_error)
                })
                .collect::<Result<Vec<_>>>()?;

            self.network.apply_gradients(&*optimizer)?;

            let costs: LayerValues = batches_forward_pass_costs.into_iter().flatten().collect();
            let ave_cost = costs.ave();
            Ok(ave_cost)
        }

        #[instrument(level = "info", skip_all)]
        fn into_batches(
            &self,
            corpus: &str,
            batch_size: TrainBatchConfig,
        ) -> Vec<(Vec<usize>, Vec<usize>)> {
            match &batch_size {
                &TrainBatchConfig::SingleBatch(batch_size)
                | &TrainBatchConfig::Batches(batch_size) => {
                    let context_len = self.context_length;
                    let mut batch_samples = vec![];
                    let corpus_len = corpus.chars().count();

                    for _ in 0..batch_size {
                        let exclusive_max = corpus_len - context_len;
                        let start_index = self.rng.rand_range(0, exclusive_max);
                        let chars = corpus
                            .chars()
                            .skip(start_index)
                            .take(context_len + 1)
                            .collect_vec();

                        batch_samples.push(chars);
                    }

                    let samples = batch_samples
                        .into_iter()
                        .filter_map(|chars| {
                            let (context, targets) =
                                (&chars[0..context_len], &chars[1..context_len + 1]);
                            Some((
                                self.encode_input_sequence(&context).ok()?,
                                self.encode_input_sequence(&targets).ok()?,
                            ))
                        })
                        .collect_vec();
                    samples
                }
            }
        }

        pub fn sample_probabilities(&self, input_sequence: &[char]) -> Result<LayerValues> {
            let network_input = self.encode_input_sequence(input_sequence)?;
            let output = self.network.forward_inference(&network_input, None, None)?;
            let probabilities = self.compute_probabilities(&output, network_input.len())?;
            Ok(probabilities)
        }

        fn compute_probabilities(
            &self,
            output: &Linear,
            network_input_len: usize,
        ) -> Result<LayerValues> {
            let output_idx = network_input_len
                .saturating_sub(1)
                .min(self.context_length - 1);
            let row: LayerValues = output.rows_iter().nth(output_idx).unwrap().into();
            self.compute_probabilities_row(&row)
        }

        fn compute_probabilities_row(&self, output: &LayerValues) -> Result<LayerValues> {
            let post_apply_mode = NetworkActivationMode::SoftMaxCrossEntropy;
            let probabilities = post_apply_mode.apply(&output);
            Ok(probabilities)
        }

        fn one_hot(&self, idx: usize) -> impl Iterator<Item = NodeValue> {
            let vocab_len = self.vocab.len();
            (0..vocab_len).map(move |i| if i == idx { 1.0 } else { 0.0 })
        }

        pub fn predict_next(&self, last_tokens: &[char]) -> Result<char> {
            let probabilities = self.sample_probabilities(last_tokens)?;
            let sampled_idx = self.rng.sample_uniform(&probabilities)?;
            let token = self.token_decode(sampled_idx);

            token.context("sampled token should be in vocab dict")
        }

        pub fn predict_from(&self, last_word: &str) -> Result<char> {
            self.predict_next(&last_word.chars().collect_vec())
        }

        pub fn predict_arg_max_from(&self, last_word: &str) -> Result<char> {
            let probabilities = self.sample_probabilities(&last_word.chars().collect_vec())?;
            let arg_max_idx = probabilities
                .position_max()
                .context("failed to perform argmax on probabilities")?;
            let token = self.token_decode(arg_max_idx);

            token.context("sampled token should be in vocab dict")
        }

        pub fn nll(&self, context_tokens: &[char], expected_next_token: char) -> Result<NodeValue> {
            let expected_encoded_token = self.token_encode(expected_next_token)?;
            let probabilities = self.sample_probabilities(context_tokens)?;
            let log_logits = probabilities
                .get(expected_encoded_token)
                .expect("output should have same count as vocab")
                .ln();

            Ok(-log_logits)
        }

        pub fn token_encode(&self, c: char) -> Result<usize> {
            self.vocab.get(&c).copied().with_context(|| {
                format!("failed to resolve vocab from dict: unknown character token '{c}'")
            })
        }

        pub fn token_decode(&self, encoded: usize) -> Option<char> {
            self.vocab
                .iter()
                .find(|(_, &vocab_token)| vocab_token == encoded)
                .map(|(&token, _)| token)
        }

        pub fn encode_input_sequence(&self, input_sequence: &[char]) -> Result<Vec<usize>> {
            let offset = input_sequence.len().saturating_sub(self.context_length);
            input_sequence
                .iter()
                .skip(offset)
                .take(self.context_length)
                .map(|&c| self.token_encode(c))
                .collect()
        }

        pub fn generate_sequence_string(
            &self,
            token_separator: &str,
            seed_token: Option<char>,
        ) -> String {
            let seed_token = seed_token.unwrap_or(CONTROL_VOCAB);
            let sequence = self
                .predict_from_iter(&[seed_token])
                .map(|c| c.to_string())
                .collect_vec();

            sequence.join(token_separator)
        }

        pub fn input_stride_width(&self) -> usize {
            self.context_length
        }

        pub fn control_vocab(&self) -> char {
            CONTROL_VOCAB
        }

        pub fn control_vocab_index(&self) -> usize {
            self.token_encode(CONTROL_VOCAB)
                .expect("CONTROL_VOCAB should be present in vocab")
        }

        pub fn predict_from_iter<'a>(
            &'a self,
            seed_tokens: &[char],
        ) -> impl Iterator<Item = char> + 'a {
            let mut token_counts = HashMap::new();
            let (curr_token, seed_tokens) = seed_tokens
                .split_last()
                .expect("should have at lease one element");

            let mut curr_token = *curr_token;
            let mut tokens_generated = 0;
            let mut context_tokens = VecDeque::from_iter(seed_tokens.iter().copied());

            let input_stride_width = self.input_stride_width();
            let max_len = 40;

            std::iter::from_fn(move || {
                if curr_token == CONTROL_VOCAB && tokens_generated > 0 {
                    None
                } else if *token_counts.get(&curr_token).unwrap_or(&0) > max_len {
                    None
                } else {
                    let last_token = curr_token.clone();

                    context_tokens.push_back(last_token.clone());
                    let skip_count = !last_token.is_alphabetic();

                    if !skip_count {
                        *token_counts.entry(last_token.clone()).or_insert(0) += 1;
                    }

                    let context_tokens_slice =
                        &context_tokens.iter().copied().collect::<Vec<_>>()[..];

                    curr_token = self.predict_next(&context_tokens_slice).ok()?;
                    tokens_generated += 1;

                    while context_tokens.len() > input_stride_width {
                        context_tokens.pop_front();
                    }

                    Some(curr_token.clone()).filter(|token| token != &CONTROL_VOCAB)
                }
            })
        }

        pub fn compute_error(&self, tokens: &[char]) -> Result<NodeValue> {
            let mut errors: Vec<NodeValue> = vec![];

            for sequence_segment in tokens.windows(self.context_length + 1) {
                let input_sequence = &sequence_segment[0..self.context_length];
                let probabilities = self.sample_probabilities(input_sequence)?;

                let target = sequence_segment[self.context_length];
                let target_token = self.token_encode(target)?;
                let expected_outputs = self.one_hot(target_token).collect();

                let train_error = probabilities.cross_entropy_error(&expected_outputs)?;
                errors.push(train_error.ave());
            }

            let errors = LayerValues::new(errors);
            Ok(errors.ave())
        }

        pub fn vocab(&self) -> &HashMap<char, usize> {
            &self.vocab
        }
    }

    #[cfg(test)]
    mod tests {
        use test_log::test;

        use crate::ml::{layer::LayerInitStrategy, transformer::solver};

        use super::*;

        #[test]
        fn transformer_work_single_simple_pattern() {
            let corpus = "abcd".to_string();
            let vocab: Vec<char> = corpus.chars().unique().collect();
            let optimizer = solver::SGDOptimizer::new_cache(0.1);
            let builder = DecoderBuilder::new(corpus.len() - 1, 12, vocab.len())
                // .with_head_count(2)
                .with_block_count(2)
                // .with_feed_forward_hidden_dimension(8)
                .with_dropout_rate(0.0);

            let transformer =
                training::setup_and_train_transformer(training::TestCharTransformerConfig {
                    builder,
                    training_rounds: 25,
                    batch_size: 1,
                    optimizer,
                    vocab,
                    training_corpus: corpus.clone(),
                    testing_corpus: corpus,
                });

            let probabilities = transformer.sample_probabilities(&['a', 'b', 'c']).unwrap();
            let token = transformer.token_encode('d').unwrap();
            let prob = dbg!(probabilities[token]);
            assert!(prob > 0.95);
        }

        #[test]
        fn transformer_works_on_partial_simple_patterns() {
            let corpus = "abcd".to_string();
            let vocab: Vec<char> = corpus.chars().unique().collect();
            let optimizer = solver::AdamOptimizer::new_cache(0.001);
            let builder = DecoderBuilder::new(3, 12, vocab.len())
                .with_block_count(6)
                .with_dropout_rate(0.0)
                .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
                // .with_output_dense_init_strategy(LayerInitStrategy::FullRandom)
                .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
                .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

            let transformer =
                training::setup_and_train_transformer(training::TestCharTransformerConfig {
                    builder,
                    training_rounds: 750,
                    batch_size: 1,
                    optimizer,
                    vocab,
                    training_corpus: corpus.clone(),
                    testing_corpus: corpus,
                });

            let probabilities = transformer.sample_probabilities(&['a', 'b']).unwrap();
            let token1 = transformer.token_encode('c').unwrap();
            let prob1 = dbg!(probabilities[token1]);
            assert!(prob1 > 0.95);
        }

        #[test]
        #[ignore = "takes too long"]
        fn transformer_works_on_simultaneous_simple_patterns() {
            let corpus = "abcdefghijklmnopqrstuvwxyz".to_string();
            let vocab: Vec<char> = corpus.chars().unique().collect();
            println!("[(0,BLOCKS),(1,ROUNDS),(2,---),(3,EMBEDIM),(4,HIDNDIM),(5,HEADS)]");
            let inject_env_values = inject_env_values();
            // let optimizer = solver::AdamOptimizer::new_cache(0.001);
            let optimizer = solver::source::DefaultOptimizerCache::new(
                solver::source::DynamicOptimizerFactory::new(
                    move |param_count, param_dimension| {
                        solver::AdamOptimizer::new_builder(param_count, param_dimension)
                            .with_beta(0.9, 0.98)
                            .with_epsilon(1e-9)
                            .build()
                    },
                ),
            );
            // let optimizer = solver::AdamOptimizer::new_cache(*inject_env_values.get(2).unwrap_or(&0.01));
            // let optimizer = solver::SGDOptimizer::new_cache(*inject_env_values.get(2).unwrap_or(&0.01));
            let model_dimension = *inject_env_values.get(3).unwrap_or(&12.0) as usize;
            let builder = DecoderBuilder::new(3, model_dimension, vocab.len())
                .with_block_count(*inject_env_values.get(0).unwrap_or(&1.0) as usize)
                .with_feed_forward_hidden_dimension(
                    *inject_env_values.get(4).unwrap_or(&64.0) as usize
                )
                .with_head_count(*inject_env_values.get(5).unwrap_or(&3.0) as usize)
                .with_dropout_rate(0.0)
                .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
                // .with_output_dense_init_strategy(LayerInitStrategy::FullRandom)
                .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
                .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

            let transformer =
                training::setup_and_train_transformer(training::TestCharTransformerConfig {
                    builder,
                    // training_rounds: 50000,
                    training_rounds: *inject_env_values.get(1).unwrap_or(&2000.0) as usize,
                    batch_size: 1,
                    optimizer,
                    vocab,
                    training_corpus: corpus.clone(),
                    testing_corpus: corpus,
                });

            // let probabilities = transformer.sample_probabilities(&['a', 'b']).unwrap();
            // let token1 = dbg!(transformer.token_encode('c').unwrap());
            // let prob1 = dbg!(probabilities[token1]);
            // assert!(prob1 > 0.95);

            let probabilities = transformer.sample_probabilities(&['a', 'b', 'c']).unwrap();
            let token1 = dbg!(transformer.token_encode('d').unwrap());
            let prob1 = dbg!(probabilities[token1]);
            assert!(prob1 > 0.95);

            let probabilities = transformer.sample_probabilities(&['b', 'c', 'd']).unwrap();
            let token2 = transformer.token_encode('e').unwrap();
            let prob2 = dbg!(probabilities[token2]);
            assert!(prob2 > 0.95);

            let probabilities = transformer.sample_probabilities(&['c', 'd', 'e']).unwrap();
            let token3 = transformer.token_encode('f').unwrap();
            let prob3 = dbg!(probabilities[token3]);
            assert!(prob3 > 0.95);
        }

        #[test]
        #[ignore = "takes too long"]
        fn transformer_work_simple_counter_prediction() {
            let phrases = [
                "six seven eight nine tho this may been seen less often",
                "zero one two three four five six seven eight nine ten",
                "cero uno dos tres cuatro cinco seis siete ocho nueve diez",
                "zero uno due tre quattro cinque sei sette otto nove dieci",
                // "nul een twee drie vier vijf zes zeven acht negen tien",
                // "null eins zwei drei vier fünf sechs sieben acht neun zehn",
                // "sero un dau tri pedwar pump chwech saith wyth naw deg",
            ];

            let corpus = phrases.join("\n");
            let vocab: Vec<char> = corpus.chars().unique().collect();
            let optimizer = solver::AdamOptimizer::new_cache(0.001);

            let builder = DecoderBuilder::new(12, 12, vocab.len())
                .with_block_count(1) // multiple blocks cause lower accuracy, check init strats and back prop grads?
                .with_dropout_rate(0.0)
                .with_embedding_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_output_dense_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_attention_kqv_weights_init_strategy(LayerInitStrategy::ScaledFullRandom)
                .with_attention_dense_layer_init_strategy(LayerInitStrategy::KaimingZeroBias)
                .with_feed_forward_init_strategy(LayerInitStrategy::KaimingZeroBias);

            let t = training::setup_and_train_transformer(training::TestCharTransformerConfig {
                builder,
                training_rounds: 10000,
                batch_size: 5,
                optimizer,
                vocab,
                training_corpus: corpus.clone(),
                testing_corpus: corpus,
            });

            let check_corpus_stride = |target: char, context: &str| {
                let probabilities = t
                    .sample_probabilities(&context.chars().collect_vec())
                    .unwrap();

                let token = t.token_encode(target).unwrap();
                let prob = probabilities[token];
                let prob_ranks = probabilities.rank_iter().collect_vec();
                let rank = prob_ranks[token];
                let prob_pct = prob * 100.0;

                println!(
                    "ctx: '{context}', target: '{target}' -> prob: {prob_pct:.2}% (rank #{rank})"
                );

                assert!(prob.is_finite());
                // assert!(prob > 0.1); // 0.95
                // assert_eq!(target, t.predict_arg_max_from(context).unwrap());
            };

            check_corpus_stride('n', "seven eight ");
            check_corpus_stride('i', "even eight n");
            check_corpus_stride('n', "ven eight ni");
            check_corpus_stride('e', "en eight nin");
            check_corpus_stride(' ', "n eight nine");

            assert_eq!('n', t.predict_arg_max_from("seven eight ").unwrap());
            assert_eq!('i', t.predict_arg_max_from("even eight n").unwrap());
            assert_eq!('n', t.predict_arg_max_from("ven eight ni").unwrap());
            assert_eq!('e', t.predict_arg_max_from("en eight nin").unwrap());
            assert_eq!(' ', t.predict_arg_max_from("n eight nine").unwrap());
        }

        fn inject_env_values() -> Vec<f64> {
            let inject_env_var = std::env::var("RUST_INJECT")
                .iter()
                .flat_map(|x| x.split(',').map(|x| x.parse::<f64>()))
                .flatten()
                .collect_vec();

            dbg!(inject_env_var)
        }

        mod training {
            use itertools::Itertools;
            use tracing::info;

            use crate::ml::{
                seq2seq::transformer::CharacterTransformer,
                transformer::{decoder::builder::DecoderBuilder, solver::source::OptimizerSource},
                NodeValue, RngStrategy,
            };

            #[derive(Clone)]
            pub struct TestCharTransformerConfig<O> {
                pub builder: DecoderBuilder,
                pub training_rounds: usize,
                pub batch_size: usize,
                pub optimizer: O,
                pub vocab: Vec<char>,
                pub training_corpus: String,
                pub testing_corpus: String,
            }

            pub fn setup_and_train_transformer<O: OptimizerSource>(
                config: TestCharTransformerConfig<O>,
            ) -> CharacterTransformer {
                let rng = RngStrategy::testable(1234);
                let builder = config.builder.with_rng(rng);
                let rng = RngStrategy::testable(12345);
                let mut transformer =
                    CharacterTransformer::from_builder_ordered(builder, &config.vocab, rng)
                        .unwrap();

                for round in 0..config.training_rounds {
                    let training_error = transformer
                        .train(
                            &config.training_corpus,
                            &config.optimizer,
                            config.batch_size,
                        )
                        .unwrap();

                    if should_report_round(round, config.training_rounds) {
                        let (validation_errors, predictions_pct) =
                            validate_test_set(&transformer, &config.testing_corpus);

                        report_training_round(
                            round,
                            training_error,
                            validation_errors,
                            predictions_pct,
                        );
                    }
                }

                transformer
            }

            fn should_report_round(round: usize, training_rounds: usize) -> bool {
                let round_1based = round + 1;

                let ten_factor = 100; //ms_per_round.map_or(100, |ms_per_round| {
                                      //     let ten_exp = (target_report_interval_ms / (ms_per_round as f64 + 1e-8))
                                      //         .log10()
                                      //         .floor()
                                      //         .max(1.0) as u32;

                //     10_usize.pow(ten_exp)
                // });

                round_1based <= 3
                    || (round_1based % ten_factor == 0)
                    || round_1based == training_rounds
            }

            fn report_training_round(
                round: usize,
                training_error: NodeValue,
                validation_errors: Vec<(NodeValue, NodeValue)>,
                predictions_pct: f64,
            ) {
                let val_count = validation_errors.len() as NodeValue;
                let (validation_error, nll) =
                    validation_errors
                        .iter()
                        .fold((0.0, 0.0), |sum, (validation_error, nll)| {
                            (
                                sum.0 + (validation_error / val_count),
                                sum.1 + (nll / val_count),
                            )
                        });

                info!(
                    "round = {:<6} |  train_loss = {:<12.10}, val_pred_acc: {:0>4.1}%, val_loss = {:<2.6e}, val_nll = {:<6.3}",
                    round + 1, training_error, predictions_pct, validation_error, nll
                );
            }

            fn validate_test_set(
                transformer: &CharacterTransformer,
                testing_phrases: &String,
            ) -> (Vec<(NodeValue, NodeValue)>, f64) {
                let testing_phrase_windows = &testing_phrases
                    .chars()
                    .chunks(transformer.input_stride_width() + 1)
                    .into_iter()
                    .map(|chunk| chunk.collect_vec())
                    .collect_vec();

                #[cfg(feature = "threadpool")]
                use rayon::prelude::*;

                let testing_phrase_windows_iter = {
                    #[cfg(feature = "threadpool")]
                    {
                        testing_phrase_windows.par_iter()
                    }
                    #[cfg(not(feature = "threadpool"))]
                    {
                        testing_phrase_windows.iter()
                    }
                };

                let (validation_errors, correct_first_word_predictions): (
                    Vec<(NodeValue, NodeValue)>,
                    Vec<usize>,
                ) = testing_phrase_windows_iter
                    .map(|testing_phrase_window| {
                        let (&last_token, context_tokens) =
                            testing_phrase_window.split_last().unwrap();

                        let predicted = transformer.predict_next(&context_tokens).unwrap();
                        let actual = last_token;
                        let correct_first_word_count = if predicted == actual { 1 } else { 0 };

                        let error = transformer
                            .compute_error(&[context_tokens, &[actual]].concat())
                            .unwrap();

                        let nll = transformer.nll(&context_tokens, actual).unwrap();
                        ((error, nll), correct_first_word_count)
                    })
                    .unzip();

                let correct_count = correct_first_word_predictions.iter().sum::<usize>();
                let predictions_pct =
                    correct_count as f64 * 100.0 / validation_errors.len() as f64;

                (validation_errors, predictions_pct)
            }
        }
    }
}
