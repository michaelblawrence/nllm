pub mod decoder {
    use anyhow::{Context, Result};
    use serde::{Deserialize, Serialize};

    use super::{
        blocks::DecoderBlock,
        dense::Dense,
        layers::{DropoutLayer, EmbeddingLayer, PositionalEmbeddingLayer},
        linear::Linear,
        solver::source::OptimizerSource,
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Decoder {
        blocks: Vec<DecoderBlock>,
        token_embedding: EmbeddingLayer,
        position_embedding: PositionalEmbeddingLayer,
        dropout: DropoutLayer,
        output_dense: Dense,
        sequence_length: usize,
        padding_token: usize,
    }

    impl Decoder {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            target_vocab_size: usize,
        ) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, target_vocab_size)
                .with_head_count(head_count)
                .build()
                .unwrap()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            target_vocab_size: usize,
        ) -> builder::DecoderBuilder {
            builder::DecoderBuilder::new(sequence_len, embedding_dimension, target_vocab_size)
        }

        pub fn forward_training<T: AsRef<[usize]>>(
            &self,
            input_sequence: T,
            encoder_output: Option<&Linear>,
            encoder_mask: Option<&Linear>,
        ) -> Result<Linear> {
            self.forward_advanced(input_sequence, encoder_output, encoder_mask, true)
        }

        pub fn forward_inference<T: AsRef<[usize]>>(
            &self,
            input_sequence: T,
            encoder_output: Option<&Linear>,
            encoder_mask: Option<&Linear>,
        ) -> Result<Linear> {
            self.forward_advanced(input_sequence, encoder_output, encoder_mask, false)
        }

        pub fn forward_advanced<T: AsRef<[usize]>>(
            &self,
            input_sequence: T,
            encoder_output: Option<&Linear>,
            encoder_mask: Option<&Linear>,
            use_dropout: bool,
        ) -> Result<Linear> {
            let input_sequence = input_sequence.as_ref();
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();

            let block_input = self.dropout.forward_if_enabled(embeddings, use_dropout)?;

            let mut block_output = block_input;
            for block in &self.blocks {
                let block_input = &block_output;
                block_output = block.forward_advanced(
                    &block_input,
                    encoder_output,
                    mask,
                    encoder_mask,
                    use_dropout,
                )?;
            }

            let output = self.output_dense.forward(&block_output)?;
            Ok(output)
        }

        pub fn backward<T: AsRef<[usize]>>(
            &self,
            input_sequence: T,
            encoder_output: Option<&Linear>,
            encoder_mask: Option<&Linear>,
            output_gradients: Linear,
        ) -> Result<Option<Linear>> {
            // forawrd pass
            let input_sequence = input_sequence.as_ref();
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = embeddings.clone(); // self.dropout.forward(&embeddings)?;

            let mut block_inputs = vec![block_input];
            for block in &self.blocks {
                let block_input = block_inputs.last().unwrap();
                let block_output = block.forward_training_with_mask(
                    &block_input,
                    encoder_output,
                    mask,
                    encoder_mask,
                )?;
                block_inputs.push(block_output);
            }
            let block_output = block_inputs.last().unwrap();

            // backward pass
            // output_gradients.show_heatmap_plot("[output_gradients]: from loss fn");
            let mut block_output_gradients = self
                .output_dense
                .backward(&block_output, &output_gradients)?;

            // block_output_gradients.show_heatmap_plot(
            //     "[block_output_gradients]: output_dense.backward(.., output_gradients)",
            // );
            // panic!();

            let mut encoder_output_grads = vec![];
            for (block, inputs) in self.blocks.iter().zip(block_inputs.iter()).rev() {
                let (block_input_gradients, encoder_output_gradients) = block.backward_with_mask(
                    inputs,
                    encoder_output,
                    mask,
                    encoder_mask,
                    &block_output_gradients,
                )?;
                block_output_gradients = block_input_gradients;
                encoder_output_grads.push(encoder_output_gradients);
            }

            let d_embeddings = block_output_gradients;
            let d_token_embeddings = &d_embeddings;
            let d_position_embeddings = &d_embeddings;

            // d_embeddings.show_heatmap_plot(
            //     "[d_embeddings]: block.backward_with_mask(.., block_output_gradients)",
            // );

            self.token_embedding
                .backward(&input_sequence, &d_token_embeddings)?;

            self.position_embedding
                .backward(0, input_sequence.len(), &d_position_embeddings)?;

            match encoder_output {
                Some(encoder_output) => Ok(Some(
                    self.sum_encoder_gradients(encoder_output, &encoder_output_grads)?,
                )),
                None => Ok(None),
            }
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            for (i, block) in self.blocks.iter_mut().enumerate() {
                block.apply_gradients(&*optimizer.with_index(i))?;
            }
            self.token_embedding
                .apply_gradients(&*optimizer.with_next_index())?;
            self.position_embedding
                .apply_gradients(&*optimizer.with_next_index())?;
            self.output_dense
                .apply_gradients(&*optimizer.with_next_index())?;

            Ok(())
        }

        fn apply_input_padding(&self, input_sequence: &[usize]) -> (Vec<usize>, Linear) {
            EmbeddingLayer::pad_inputs(input_sequence, self.sequence_length, self.padding_token)
        }

        fn sum_encoder_gradients(
            &self,
            encoder_outputs: &Linear,
            encoder_gradients: &Vec<Option<Linear>>,
        ) -> Result<Linear> {
            let zero: Linear = Linear::with_dimensions(encoder_outputs);
            let mut sum_outputs_grads = zero.iter().boxed();
            for outputs_grads in encoder_gradients.iter() {
                let grads = outputs_grads
                    .as_ref()
                    .context("missing gradient for encoder")?;

                sum_outputs_grads = sum_outputs_grads.add(grads.iter()).boxed();
            }
            Ok(sum_outputs_grads.collect())
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::{
            lazy_opt,
            ml::{
                transformer::{
                    solver::{self, source::DynamicOptimizerFactory, Optimizer},
                    tests::helpers::{assert_optimisation_converges, new_linear},
                },
                NodeValue, RngStrategy,
            },
        };

        use super::*;

        #[test]
        fn decoder_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let decoder = Decoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            let output = decoder.forward_training(&inputs, None, None).unwrap();
            println!("outputs -> {output:#}");

            let outputs = output.clone().to_values();
            assert_eq!(outputs.len(), seq_len);
            assert_eq!(outputs[0].len(), vocab_size);
            assert!(output.is_finite());

            let max_output_value = outputs
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| x.abs())
                .reduce(NodeValue::max)
                .unwrap();
            let max_expected_output_value = 3.2;

            assert_eq!(
                max_output_value.max(max_expected_output_value),
                max_expected_output_value,
                "initial state is not diffuse"
            );
        }

        #[test]
        fn decoder_can_process_inputs_shorter_than_seq_len() {
            let seq_len = 4;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let decoder = Decoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            assert!(inputs.len() < seq_len);

            let output = decoder.forward_training(&inputs, None, None).unwrap();
            println!("outputs -> {output:#}");

            let outputs = output.clone().to_values();
            assert_eq!(outputs.len(), seq_len);
            assert_eq!(outputs[0].len(), vocab_size);
            assert!(output.is_finite());
        }

        #[test]
        fn decoder_can_minimise() {
            let seq_len = 3;
            let embed_dim = 6;
            let head_count = 2;
            let hidden_dim = 8;
            let vocab_size = 10;
            let block_count = 2;

            let learn_rate = 0.01;
            let iters = 25;
            let dummy_grads = Linear::new(1, 1);
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let decoder = Decoder::new_builder(seq_len, embed_dim, vocab_size)
                        .with_head_count(head_count)
                        .with_block_count(block_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .with_dropout_rate(0.0) // TODO: reenable dropout once relate TODOs are completed
                        .build()
                        .unwrap();
                    let inputs = [3, 6, 9];
                    let target = new_linear(seq_len, vocab_size, &rng);
                    (decoder, inputs, target)
                },
                &move |decoder, inputs| decoder.forward_training(&inputs, None, None).unwrap(),
                &move |decoder, inputs, dloss| {
                    decoder.backward(&inputs, None, None, dloss).unwrap();
                    decoder.apply_gradients(&optimizer).unwrap();
                    dummy_grads.clone()
                },
                iters,
            );
        }

        #[test]
        fn decoder_can_minimise_with_encoder_outputs() {
            let seq_len = 3;
            let embed_dim = 6;
            let head_count = 2;
            let hidden_dim = 8;
            let vocab_size = 10;
            let block_count = 6;

            let optimizer = solver::SGDOptimizer::new_cache(0.001);
            let iters = 25;

            assert_optimisation_converges(
                &move |rng| {
                    let decoder = Decoder::new_builder(seq_len, embed_dim, vocab_size)
                        .with_head_count(head_count)
                        .with_block_count(block_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .with_dropout_rate(0.0) // TODO: reenable dropout once relate TODOs are completed
                        .build()
                        .unwrap();
                    let inputs = [3, 6, 9];
                    let target = new_linear(seq_len, vocab_size, &rng);
                    let encoder_outputs = new_linear(seq_len, embed_dim, &rng);
                    (decoder, (inputs, encoder_outputs), target)
                },
                &move |decoder, (inputs, encoder_outputs)| {
                    decoder
                        .forward_training(&inputs, Some(encoder_outputs), None)
                        .unwrap()
                },
                &move |decoder, (inputs, encoder_outputs), dloss| {
                    let encoder_grads = decoder
                        .backward(&inputs, Some(encoder_outputs), None, dloss)
                        .unwrap()
                        .unwrap();

                    let mut opt = lazy_opt!(optimizer);
                    decoder.apply_gradients(&optimizer).unwrap();
                    opt.update(encoder_outputs, &encoder_grads).unwrap();
                    encoder_grads
                },
                iters,
            );
        }

        #[test]
        #[ignore = "evaluating optimization algorithms, takes long"]
        fn decoder_can_minimise_big_models_evaluating_opt_algos() {
            fn run_cycle<T: OptimizerSource>(label: &str, optimizer: T) {
                let seq_len = 128;
                let embed_dim = 32;
                let head_count = 4;
                let hidden_dim = 48;
                let vocab_size = 100;
                let block_count = 6;

                let iters = 100;
                assert_optimisation_converges(
                    &move |rng| {
                        println!("Creating instance for optimizer test = '{label}'");
                        // let rng = RngStrategy::default();
                        let decoder = Decoder::new_builder(seq_len, embed_dim, vocab_size)
                            .with_head_count(head_count)
                            .with_block_count(block_count)
                            .with_feed_forward_hidden_dimension(hidden_dim)
                            .with_rng(rng.clone())
                            .with_dropout_rate(0.0) // TODO: reenable dropout once relate TODOs are completed
                            .build()
                            .unwrap();
                        let inputs = [3, 6, 9];
                        let target = new_linear(seq_len, vocab_size, &rng);
                        (decoder, inputs, target)
                    },
                    &move |decoder, inputs| decoder.forward_training(&inputs, None, None).unwrap(),
                    &move |decoder, inputs, dloss| {
                        decoder.backward(&inputs, None, None, dloss).unwrap();
                        decoder.apply_gradients(&optimizer).unwrap();
                        Linear::new(1, 1)
                    },
                    iters,
                );
            }

            run_cycle("SGDOptimizer", solver::SGDOptimizer::new_cache(0.0001));
            run_cycle("AdamOptimizer", solver::AdamOptimizer::new_cache(0.01));
            run_cycle(
                "AdamOptimizerTuned",
                solver::source::DefaultOptimizerCache::new(DynamicOptimizerFactory::new(
                    |param_count, param_dimension| {
                        solver::AdamOptimizer::new_builder(param_count, param_dimension)
                            .with_eta(0.01)
                            .with_beta(0.89, 0.995)
                            .build()
                    },
                )),
            );
            run_cycle(
                "RMSpropOptimizerTuned",
                solver::source::DefaultOptimizerCache::new(DynamicOptimizerFactory::new(
                    |param_count, param_dimension| {
                        solver::RMSpropOptimizer::new(param_count, param_dimension)
                            .with_eta(0.01)
                            .with_gamma(0.995)
                    },
                )),
            );
            run_cycle(
                "RMSpropOptimizerTuned_opt_gamma-0.999",
                solver::source::DefaultOptimizerCache::new(DynamicOptimizerFactory::new(
                    |param_count, param_dimension| {
                        solver::RMSpropOptimizer::new(param_count, param_dimension)
                            .with_eta(0.01)
                            .with_gamma(0.999)
                    },
                )),
            );
            run_cycle(
                "SGDOptimizer_rate-0.0002",
                solver::SGDOptimizer::new_cache(0.0002),
            );
        }
    }

    pub mod builder {
        use anyhow::{anyhow, Result};

        use super::{super::layers::DropoutLayer, Decoder};

        use crate::ml::{
            layer::LayerInitStrategy,
            transformer::{
                blocks::DecoderBlock,
                dense::Dense,
                layers::{EmbeddingLayer, PositionalEmbeddingLayer},
            },
            NodeValue, RngStrategy,
        };

        #[derive(Clone)]
        pub struct DecoderBuilder {
            sequence_len: usize,
            model_dimension: usize,
            head_count: Option<usize>,
            target_vocab_size: usize,
            block_count: usize,
            padding_token: usize,
            rng: RngStrategy,
            embedding_init_strategy: LayerInitStrategy,
            output_dense_init_strategy: LayerInitStrategy,
            attention_kqv_weights_init_strategy: LayerInitStrategy,
            attention_dense_layer_init_strategy: LayerInitStrategy,
            feed_forward_init_strategy: LayerInitStrategy,
            feed_forward_hidden_dimension: usize,
            dropout_rate: NodeValue,
        }

        impl DecoderBuilder {
            pub fn new(
                sequence_len: usize,
                model_dimension: usize,
                target_vocab_size: usize,
            ) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count: None,
                    target_vocab_size,
                    block_count: 6,
                    padding_token: 0,
                    rng: Default::default(),
                    embedding_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    output_dense_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    attention_kqv_weights_init_strategy: LayerInitStrategy::ScaledFullRandom,
                    attention_dense_layer_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            fn get_default_head_count(&self) -> Option<usize> {
                let search_candidates = vec![3, 4, 5];

                search_candidates
                    .into_iter()
                    .find(|head_count| self.model_dimension % head_count == 0)
            }

            pub fn build(self) -> Result<Decoder> {
                let head_count = match self.head_count.or_else(|| self.get_default_head_count()) {
                    Some(head_count) if self.model_dimension % head_count == 0 => head_count,
                    _ => Err(anyhow!(
                        "head_count is not a factor of model_dimension={}",
                        self.model_dimension
                    ))?,
                };
                if self.padding_token >= self.target_vocab_size {
                    Err(anyhow!(
                        "padding_token index is not within the target_vocab_size dimension"
                    ))?;
                }
                let block_builder = DecoderBlock::new_builder(
                    self.sequence_len,
                    self.model_dimension,
                    head_count,
                    head_count,
                );
                let block_builder = block_builder
                    .with_rng(self.rng.clone())
                    .with_dropout_rate(self.dropout_rate)
                    .with_attention_kqv_weights_init_strategy(
                        self.attention_kqv_weights_init_strategy,
                    )
                    .with_attention_dense_layer_init_strategy(
                        self.attention_dense_layer_init_strategy,
                    )
                    .with_feed_forward_hidden_dimension(self.feed_forward_hidden_dimension)
                    .with_feed_forward_init_strategy(self.feed_forward_init_strategy);

                let mut blocks = vec![];
                for _ in 0..self.block_count {
                    let block_builder = block_builder.clone();
                    let block = block_builder.build()?;
                    blocks.push(block);
                }
                let token_embedding = EmbeddingLayer::new(
                    self.model_dimension,
                    self.target_vocab_size,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let position_embedding = PositionalEmbeddingLayer::new(
                    self.model_dimension,
                    self.sequence_len,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let dropout = DropoutLayer::new(self.dropout_rate, &self.rng);
                let output_dense = Dense::new(
                    self.model_dimension,
                    self.target_vocab_size,
                    &self.output_dense_init_strategy,
                    &self.rng,
                );

                Ok(Decoder {
                    blocks,
                    token_embedding,
                    position_embedding,
                    dropout,
                    output_dense,
                    sequence_length: self.sequence_len,
                    padding_token: self.padding_token,
                })
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_target_vocab_size(mut self, target_vocab_size: usize) -> Self {
                self.target_vocab_size = target_vocab_size;
                self
            }

            pub fn with_embedding_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.embedding_init_strategy = init_strategy;
                self
            }

            pub fn with_output_dense_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.output_dense_init_strategy = init_strategy;
                self
            }

            pub fn with_attention_kqv_weights_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.attention_kqv_weights_init_strategy = init_strategy;
                self
            }

            pub fn with_attention_dense_layer_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.attention_dense_layer_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.feed_forward_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_hidden_dimension(mut self, hidden_dimension: usize) -> Self {
                self.feed_forward_hidden_dimension = hidden_dimension;
                self
            }

            pub fn with_dropout_rate(mut self, dropout_rate: NodeValue) -> Self {
                self.dropout_rate = dropout_rate;
                self
            }

            pub fn with_block_count(mut self, block_count: usize) -> Self {
                self.block_count = block_count;
                self
            }

            pub fn with_head_count(mut self, head_count: usize) -> Self {
                self.head_count = Some(head_count);
                self
            }

            pub fn with_padding_token(mut self, padding_token: usize) -> Self {
                self.padding_token = padding_token;
                self
            }

            pub fn rng(&self) -> &RngStrategy {
                &self.rng
            }

            pub fn sequence_len(&self) -> usize {
                self.sequence_len
            }
        }
    }
}

pub mod encoder {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    use super::{
        blocks::EncoderBlock,
        layers::{DropoutLayer, EmbeddingLayer, PositionalEmbeddingLayer},
        linear::Linear,
        solver::source::OptimizerSource,
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Encoder {
        blocks: Vec<EncoderBlock>,
        token_embedding: EmbeddingLayer,
        position_embedding: PositionalEmbeddingLayer,
        dropout: DropoutLayer,
        sequence_length: usize,
        padding_token: usize,
    }

    impl Encoder {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            source_vocab_size: usize,
        ) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, source_vocab_size)
                .with_head_count(head_count)
                .build()
                .unwrap()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            source_vocab_size: usize,
        ) -> builder::EncoderBuilder {
            builder::EncoderBuilder::new(sequence_len, embedding_dimension, source_vocab_size)
        }

        pub fn forward_training<T: AsRef<[usize]>>(&self, input_sequence: T) -> Result<Linear> {
            let input_sequence = input_sequence.as_ref();
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = self.dropout.forward(&embeddings)?;

            let mut block_output = block_input;
            for block in &self.blocks {
                let block_input = &block_output;
                block_output = block.forward_training_with_mask(&block_input, mask)?;
            }

            Ok(block_output)
        }

        pub fn backward<T: AsRef<[usize]>>(
            &mut self,
            input_sequence: T,
            output_gradients: Linear,
        ) -> Result<()> {
            // forawrd pass
            let input_sequence = input_sequence.as_ref();
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = embeddings.clone(); // self.dropout.forward(&embeddings)?;

            let mut block_inputs = vec![block_input];
            for block in &self.blocks {
                let block_input = block_inputs.last().unwrap();
                let block_output = block.forward_training_with_mask(&block_input, mask)?;
                block_inputs.push(block_output);
            }

            // backward pass
            let mut block_output_gradients = output_gradients;
            for (block, inputs) in self.blocks.iter_mut().zip(block_inputs.iter()).rev() {
                let block_input_gradients =
                    block.backward_with_mask(&inputs, mask, &block_output_gradients)?;
                block_output_gradients = block_input_gradients;
            }

            let d_embeddings = block_output_gradients;
            let d_token_embeddings = &d_embeddings;
            let d_position_embeddings = &d_embeddings;

            self.token_embedding
                .backward(&input_sequence, &d_token_embeddings)?;

            self.position_embedding
                .backward(0, input_sequence.len(), &d_position_embeddings)?;

            Ok(())
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            for block in self.blocks.iter_mut() {
                block.apply_gradients(optimizer)?;
            }
            self.token_embedding.apply_gradients(optimizer)?;
            self.position_embedding.apply_gradients(optimizer)?;

            Ok(())
        }

        fn apply_input_padding(&self, input_sequence: &[usize]) -> (Vec<usize>, Linear) {
            EmbeddingLayer::pad_inputs(input_sequence, self.sequence_length, self.padding_token)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::{
                solver,
                tests::helpers::{assert_optimisation_converges, new_linear},
            },
            NodeValue, RngStrategy,
        };

        use super::*;

        #[test]
        fn encoder_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            let output = encoder.forward_training(&inputs).unwrap();
            println!("outputs -> {output:#}");

            let outputs = output.clone().to_values();
            assert_eq!(outputs.len(), seq_len);
            assert_eq!(outputs[0].len(), embed_dim);
            assert!(output.is_finite());

            let max_output_value = outputs
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| x.abs())
                .reduce(NodeValue::max)
                .unwrap();
            let max_expected_output_value = 3.2;

            assert_eq!(
                max_output_value.max(max_expected_output_value),
                max_expected_output_value,
                "initial state is not diffuse"
            );
        }

        #[test]
        fn encoder_can_process_inputs_shorter_than_seq_len() {
            let seq_len = 4;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            assert!(inputs.len() < seq_len);

            let output = encoder.forward_training(&inputs).unwrap();
            println!("outputs -> {output:#}");

            let outputs = output.clone().to_values();
            assert_eq!(outputs.len(), seq_len);
            assert_eq!(outputs[0].len(), embed_dim);
            assert!(output.is_finite());
        }

        #[test]
        fn encoder_can_minimise() {
            let seq_len = 3;
            let embed_dim = 6;
            let head_count = 2;
            let hidden_dim = 8;
            let vocab_size = 10;
            let block_count = 6;

            let learn_rate = 0.01;
            let iters = 25;
            let dummy_grads = Linear::new(1, 1);
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                        .with_head_count(head_count)
                        .with_block_count(block_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .with_dropout_rate(0.0) // TODO: reenable dropout once relate TODOs are completed
                        .build()
                        .unwrap();
                    let inputs = [3, 6, 9];
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (encoder, inputs, target)
                },
                &move |encoder, inputs| encoder.forward_training(&inputs).unwrap(),
                &move |encoder, inputs, dloss| {
                    encoder.backward(&inputs, dloss).unwrap();
                    encoder.apply_gradients(&optimizer).unwrap();
                    dummy_grads.clone()
                },
                iters,
            );
        }
    }

    pub mod builder {
        use anyhow::{anyhow, Result};

        use super::{super::layers::DropoutLayer, Encoder};

        use crate::ml::{
            layer::LayerInitStrategy,
            transformer::{
                blocks::EncoderBlock,
                layers::{EmbeddingLayer, PositionalEmbeddingLayer},
            },
            NodeValue, RngStrategy,
        };

        pub struct EncoderBuilder {
            sequence_len: usize,
            model_dimension: usize,
            head_count: usize,
            source_vocab_size: usize,
            block_count: usize,
            padding_token: usize,
            rng: RngStrategy,
            embedding_init_strategy: LayerInitStrategy,
            feed_forward_init_strategy: LayerInitStrategy,
            feed_forward_hidden_dimension: usize,
            dropout_rate: NodeValue,
        }

        impl EncoderBuilder {
            pub fn new(
                sequence_len: usize,
                model_dimension: usize,
                source_vocab_size: usize,
            ) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count: 3,
                    source_vocab_size,
                    block_count: 6,
                    padding_token: 0,
                    rng: Default::default(),
                    embedding_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> Result<Encoder> {
                if self.model_dimension % self.block_count != 0 {
                    Err(anyhow!("block_count is not a factor of model_dimension"))?;
                }
                let block_builder = EncoderBlock::new_builder(
                    self.sequence_len,
                    self.model_dimension,
                    self.head_count,
                );
                let block_builder = block_builder
                    .with_rng(self.rng.clone())
                    .with_dropout_rate(self.dropout_rate)
                    .with_feed_forward_hidden_dimension(self.feed_forward_hidden_dimension)
                    .with_feed_forward_init_strategy(self.feed_forward_init_strategy);

                let mut blocks = vec![];
                for _ in 0..self.block_count {
                    let block = block_builder.clone().build();
                    blocks.push(block);
                }
                let token_embedding = EmbeddingLayer::new(
                    self.model_dimension,
                    self.source_vocab_size,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let position_embedding = PositionalEmbeddingLayer::new(
                    self.model_dimension,
                    self.sequence_len,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let dropout = DropoutLayer::new(self.dropout_rate, &self.rng);

                Ok(Encoder {
                    blocks,
                    token_embedding,
                    position_embedding,
                    dropout,
                    sequence_length: self.sequence_len,
                    padding_token: self.padding_token,
                })
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_embedding_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.embedding_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.feed_forward_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_hidden_dimension(mut self, hidden_dimension: usize) -> Self {
                self.feed_forward_hidden_dimension = hidden_dimension;
                self
            }

            pub fn with_dropout_rate(mut self, dropout_rate: NodeValue) -> Self {
                self.dropout_rate = dropout_rate;
                self
            }

            pub fn with_block_count(mut self, block_count: usize) -> Self {
                self.block_count = block_count;
                self
            }

            pub fn with_head_count(mut self, head_count: usize) -> Self {
                self.head_count = head_count;
                self
            }

            pub fn with_padding_token(mut self, padding_token: usize) -> Self {
                self.padding_token = padding_token;
                self
            }
        }
    }
}

pub mod blocks {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    use super::{
        layers::{
            DropoutLayer, FeedForwardLayer, LayerNormalization, MultiHeadCrossAttentionLayer,
            MultiHeadSelfAttentionLayer,
        },
        linear::Linear,
        solver::source::OptimizerSource,
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecoderBlock {
        masked_self_attention: MultiHeadSelfAttentionLayer,
        encoder_attention: MultiHeadCrossAttentionLayer,
        network: FeedForwardLayer,
        dropout: (DropoutLayer, DropoutLayer, DropoutLayer),
        layer_norm: (LayerNormalization, LayerNormalization, LayerNormalization),
    }

    impl DecoderBlock {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            cross_attention_head_count: usize,
        ) -> Result<Self> {
            Self::new_builder(
                sequence_len,
                embedding_dimension,
                head_count,
                cross_attention_head_count,
            )
            .build()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            cross_attention_head_count: usize,
        ) -> builder::DecoderBlockBuilder {
            builder::DecoderBlockBuilder::new(
                sequence_len,
                embedding_dimension,
                head_count,
                cross_attention_head_count,
            )
        }

        pub fn forward_training(
            &self,
            inputs: &Linear,
            encoder_output: Option<&Linear>,
        ) -> Result<Linear> {
            self.forward_training_with_mask(inputs, encoder_output, None, None)
        }

        pub fn forward_training_with_mask(
            &self,
            inputs: &Linear,
            encoder_output: Option<&Linear>,
            mask: Option<&Linear>,
            encoder_mask: Option<&Linear>,
        ) -> Result<Linear> {
            self.forward_advanced(inputs, encoder_output, mask, encoder_mask, true)
        }

        pub fn forward_advanced(
            &self,
            inputs: &Linear,
            encoder_output: Option<&Linear>,
            mask: Option<&Linear>,
            encoder_mask: Option<&Linear>,
            use_dropout: bool,
        ) -> Result<Linear> {
            let self_attention_output = self
                .masked_self_attention
                .forward_with_mask(&inputs, mask)?;
            let self_attention_output = self
                .dropout
                .0
                .forward_if_enabled(self_attention_output, use_dropout)?;

            let skip_self_attention_output =
                self_attention_output.iter().add(inputs.iter()).collect();
            let decoder_attention_inputs =
                self.layer_norm.0.forward(&skip_self_attention_output)?;

            let ff_network_inputs = match encoder_output {
                Some(encoder_output) => {
                    let encoder_attention_output = self.encoder_attention.forward_with_mask(
                        &encoder_output,
                        &decoder_attention_inputs,
                        encoder_mask,
                    )?;
                    let encoder_attention_output = self
                        .dropout
                        .1
                        .forward_if_enabled(encoder_attention_output, use_dropout)?;

                    let skip_encoder_attention_output = encoder_attention_output
                        .iter()
                        .add(decoder_attention_inputs.iter())
                        .collect();

                    self.layer_norm.1.forward(&skip_encoder_attention_output)?
                }
                None => decoder_attention_inputs,
            };

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let ff_network_output = self
                .dropout
                .2
                .forward_if_enabled(ff_network_output, use_dropout)?;

            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();
            let decoder_output = self.layer_norm.2.forward(&skip_ff_output)?;

            Ok(decoder_output)
        }

        pub fn backward(
            &mut self,
            inputs: &Linear,
            encoder_output: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Option<Linear>)> {
            self.backward_with_mask(inputs, encoder_output, None, None, output_gradients)
        }

        pub fn backward_with_mask(
            &self,
            inputs: &Linear,
            encoder_output: Option<&Linear>,
            mask: Option<&Linear>,
            encoder_mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Option<Linear>)> {
            // forward pass
            let self_attention_output = self
                .masked_self_attention
                .forward_with_mask(&inputs, mask)?;
            let skip_self_attention_output =
                self_attention_output.iter().add(inputs.iter()).collect();
            let decoder_attention_inputs =
                self.layer_norm.0.forward(&skip_self_attention_output)?;

            let ff_network_inputs = match encoder_output {
                Some(encoder_output) => {
                    let encoder_attention_output = self.encoder_attention.forward_with_mask(
                        &encoder_output,
                        &decoder_attention_inputs,
                        encoder_mask,
                    )?;
                    let skip_encoder_attention_output = encoder_attention_output
                        .iter()
                        .add(decoder_attention_inputs.iter())
                        .collect();
                    self.layer_norm.1.forward(&skip_encoder_attention_output)?
                }
                None => decoder_attention_inputs.clone(),
            };

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();

            // backward pass
            let d_skip_ff_output = self
                .layer_norm
                .2
                .backward(&skip_ff_output, &output_gradients)?;

            // TODO: add dropouts backward pass
            let d_ff_network_output = &d_skip_ff_output;

            let ff_gradients = self
                .network
                .backward(&ff_network_inputs, &d_ff_network_output)?;

            let d_ff_network_inputs = d_skip_ff_output.iter().add(ff_gradients.iter()).collect();

            let (d_decoder_attention_inputs, d_encoder_output) = match encoder_output {
                Some(encoder_output) => {
                    // forward pass
                    let encoder_attention_output = self.encoder_attention.forward_with_mask(
                        &encoder_output,
                        &decoder_attention_inputs,
                        encoder_mask,
                    )?;
                    let skip_encoder_attention_output = encoder_attention_output
                        .iter()
                        .add(decoder_attention_inputs.iter())
                        .collect();

                    let d_skip_encoder_attention_output = self
                        .layer_norm
                        .1
                        .backward(&skip_encoder_attention_output, &d_ff_network_inputs)?;
                    let d_encoder_attention_output = &d_skip_encoder_attention_output;
                    let encoder_kv_q_attention_grads = self.encoder_attention.backward_with_mask(
                        &encoder_output,
                        &decoder_attention_inputs,
                        encoder_mask,
                        d_encoder_attention_output,
                    )?;

                    let (d_encoder_output, attention_gradients) = encoder_kv_q_attention_grads;
                    (
                        d_skip_encoder_attention_output
                            .iter()
                            .add(attention_gradients.iter())
                            .collect(),
                        Some(d_encoder_output),
                    )
                }
                None => (d_ff_network_inputs, None),
            };

            let d_skip_self_attention_output = self
                .layer_norm
                .0
                .backward(&skip_self_attention_output, &d_decoder_attention_inputs)?;

            let d_self_attention_output = &d_skip_self_attention_output;

            let attention_gradients = self.masked_self_attention.backward_with_mask(
                &inputs,
                mask,
                &d_self_attention_output,
            )?;

            let d_inputs = d_skip_self_attention_output
                .iter()
                .add(attention_gradients.iter())
                .collect();

            Ok((d_inputs, d_encoder_output))
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.masked_self_attention
                .apply_gradients(&*optimizer.with_index(1))?;
            self.encoder_attention
                .apply_gradients(&*optimizer.with_index(2))?;
            self.network
                .apply_gradients(&*optimizer.with_next_index())?;
            self.layer_norm
                .0
                .apply_gradients(&*optimizer.with_index(1))?;
            self.layer_norm
                .1
                .apply_gradients(&*optimizer.with_index(2))?;
            self.layer_norm
                .2
                .apply_gradients(&*optimizer.with_index(3))?;

            Ok(())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EncoderBlock {
        attention: MultiHeadSelfAttentionLayer,
        network: FeedForwardLayer,
        dropout: (DropoutLayer, DropoutLayer),
        layer_norm: (LayerNormalization, LayerNormalization),
    }

    impl EncoderBlock {
        pub fn new(sequence_len: usize, embedding_dimension: usize, head_count: usize) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, head_count).build()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
        ) -> builder::EncoderBlockBuilder {
            builder::EncoderBlockBuilder::new(sequence_len, embedding_dimension, head_count)
        }

        pub fn forward_training(&self, inputs: &Linear) -> Result<Linear> {
            self.forward_training_with_mask(inputs, None)
        }

        pub fn forward_training_with_mask(
            &self,
            inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let attention_output = self.attention.forward_with_mask(&inputs, mask)?;
            let attention_output = self.dropout.0.forward(&attention_output)?;

            let skip_attention_output = attention_output.iter().add(inputs.iter()).collect();
            let ff_network_inputs = self.layer_norm.0.forward(&skip_attention_output)?;

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let ff_network_output = self.dropout.1.forward(&ff_network_output)?;

            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();
            let encoder_output = self.layer_norm.1.forward(&skip_ff_output)?;

            Ok(encoder_output)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            self.backward_with_mask(inputs, None, output_gradients)
        }

        pub fn backward_with_mask(
            &mut self,
            inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Linear> {
            // forward pass
            let attention_output = self.attention.forward_with_mask(&inputs, mask)?;
            let skip_attention_output = attention_output.iter().add(inputs.iter()).collect();
            let ff_network_inputs = self.layer_norm.0.forward(&skip_attention_output)?;

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();

            // backward pass
            let d_skip_ff_output = self
                .layer_norm
                .1
                .backward(&skip_ff_output, &output_gradients)?;

            // TODO: add dropouts backward pass
            let d_ff_network_output = &d_skip_ff_output;

            let ff_gradients = self
                .network
                .backward(&ff_network_inputs, &d_ff_network_output)?;

            let d_ff_network_inputs = d_skip_ff_output.iter().add(ff_gradients.iter()).collect();
            let d_skip_attention_output = self
                .layer_norm
                .0
                .backward(&skip_attention_output, &d_ff_network_inputs)?;
            let d_attention_output = &d_skip_attention_output;

            let attention_gradients =
                self.attention
                    .backward_with_mask(&inputs, mask, &d_attention_output)?;

            let d_inputs = d_skip_attention_output
                .iter()
                .add(attention_gradients.iter())
                .collect();

            Ok(d_inputs)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.attention.apply_gradients(optimizer)?;
            self.network.apply_gradients(optimizer)?;
            self.layer_norm.0.apply_gradients(optimizer)?;
            self.layer_norm.1.apply_gradients(optimizer)?;

            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::{
                solver,
                tests::helpers::{
                    assert_input_gradients, assert_optimisation_converges, new_linear,
                },
            },
            RngStrategy,
        };

        use super::*;

        #[test]
        fn encoder_block_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;

            let rng = RngStrategy::testable(1234);

            let encoder_block = EncoderBlock::new_builder(seq_len, embed_dim, head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng.clone())
                .build();

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = encoder_block.forward_training(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn encoder_block_can_optimise() {
            let seq_len = 3;
            let embed_dim = 8;
            let head_count = 2;
            let hidden_dim = 24;

            let learn_rate = 0.01;
            let iters = 25;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let encoder_block = EncoderBlock::new_builder(seq_len, embed_dim, head_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .build();
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (encoder_block, inputs, target)
                },
                &move |encoder_block, inputs| encoder_block.forward_training(&inputs).unwrap(),
                &move |encoder_block, inputs, dloss| {
                    let grads = encoder_block.backward(&inputs, &dloss).unwrap();
                    encoder_block.apply_gradients(&optimizer).unwrap();
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
        }

        #[test]
        fn encoder_block_can_compute_valid_gradients() {
            let seq_len = 3;
            let embed_dim = 8;
            let head_count = 2;
            let hidden_dim = 24;

            assert_input_gradients(
                &move |rng| {
                    let encoder_block = EncoderBlock::new_builder(seq_len, embed_dim, head_count)
                        .with_dropout_rate(0.0)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .build();
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (encoder_block, inputs, target)
                },
                &move |encoder_block, inputs| encoder_block.forward_training(&inputs).unwrap(),
                &move |encoder_block, inputs, dloss| {
                    encoder_block.backward(&inputs, &dloss).unwrap()
                },
            );
        }

        #[test]
        fn decoder_block_can_compute_valid_gradients() {
            let seq_len = 3;
            let embed_dim = 8;
            let head_count = 2;
            let hidden_dim = 24;

            assert_input_gradients(
                &move |rng| {
                    let decoder_block =
                        DecoderBlock::new_builder(seq_len, embed_dim, head_count, head_count)
                            .with_dropout_rate(0.0)
                            .with_feed_forward_hidden_dimension(hidden_dim)
                            .with_rng(rng.clone())
                            .build()
                            .unwrap();
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (decoder_block, inputs, target)
                },
                &move |decoder_block, inputs| {
                    decoder_block.forward_training(&inputs, None).unwrap()
                },
                &move |decoder_block, inputs, dloss| {
                    decoder_block.backward(&inputs, None, &dloss).unwrap().0
                },
            );
        }
    }

    mod builder {
        use anyhow::Result;

        use super::{
            super::layers::{DropoutLayer, FeedForwardLayer, LayerNormalization},
            DecoderBlock, EncoderBlock,
        };

        use crate::ml::{
            layer::LayerInitStrategy,
            transformer::{
                layers::{MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer},
                linear::Linear,
            },
            NodeValue, RngStrategy,
        };

        #[derive(Debug, Clone)]
        pub struct DecoderBlockBuilder {
            sequence_len: usize,
            model_dimension: usize,
            head_count: usize,
            cross_attention_head_count: usize,
            rng: RngStrategy,
            attention_kqv_weights_init_strategy: LayerInitStrategy,
            attention_dense_layer_init_strategy: LayerInitStrategy,
            feed_forward_init_strategy: LayerInitStrategy,
            feed_forward_hidden_dimension: usize,
            dropout_rate: NodeValue,
        }

        impl DecoderBlockBuilder {
            pub fn new(
                sequence_len: usize,
                model_dimension: usize,
                head_count: usize,
                cross_attention_head_count: usize,
            ) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count,
                    cross_attention_head_count,
                    rng: Default::default(),
                    attention_kqv_weights_init_strategy: LayerInitStrategy::FullRandom,
                    attention_dense_layer_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> Result<DecoderBlock> {
                let self_attention_mask = Linear::identity(self.sequence_len);
                let masked_self_attention = MultiHeadSelfAttentionLayer::new(
                    self.sequence_len,
                    self.model_dimension,
                    self.head_count,
                    &self.attention_kqv_weights_init_strategy,
                    &self.attention_dense_layer_init_strategy,
                    &self.rng,
                )
                .with_mask(self_attention_mask);
                let encoder_attention = MultiHeadCrossAttentionLayer::new(
                    self.sequence_len,
                    self.model_dimension,
                    self.cross_attention_head_count,
                    &self.attention_kqv_weights_init_strategy,
                    &self.attention_dense_layer_init_strategy,
                    &self.rng,
                );
                let network = FeedForwardLayer::new(
                    self.model_dimension,
                    self.feed_forward_hidden_dimension,
                    &self.feed_forward_init_strategy,
                    &self.rng,
                );
                let dropout1 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let dropout2 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let dropout3 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let layer_norm1 = LayerNormalization::new(self.sequence_len);
                let layer_norm2 = LayerNormalization::new(self.sequence_len);
                let layer_norm3 = LayerNormalization::new(self.sequence_len);

                Ok(DecoderBlock {
                    masked_self_attention,
                    encoder_attention,
                    network,
                    dropout: (dropout1, dropout2, dropout3),
                    layer_norm: (layer_norm1, layer_norm2, layer_norm3),
                })
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_attention_kqv_weights_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.attention_kqv_weights_init_strategy = init_strategy;
                self
            }

            pub fn with_attention_dense_layer_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.attention_dense_layer_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.feed_forward_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_hidden_dimension(mut self, hidden_dimension: usize) -> Self {
                self.feed_forward_hidden_dimension = hidden_dimension;
                self
            }

            pub fn with_dropout_rate(mut self, dropout_rate: NodeValue) -> Self {
                self.dropout_rate = dropout_rate;
                self
            }
        }

        #[derive(Debug, Clone)]
        pub struct EncoderBlockBuilder {
            sequence_len: usize,
            model_dimension: usize,
            head_count: usize,
            rng: RngStrategy,
            attention_kqv_weights_init_strategy: LayerInitStrategy,
            attention_dense_layer_init_strategy: LayerInitStrategy,
            feed_forward_init_strategy: LayerInitStrategy,
            feed_forward_hidden_dimension: usize,
            dropout_rate: NodeValue,
        }

        impl EncoderBlockBuilder {
            pub fn new(sequence_len: usize, model_dimension: usize, head_count: usize) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count,
                    rng: Default::default(),
                    attention_kqv_weights_init_strategy: LayerInitStrategy::FullRandom,
                    attention_dense_layer_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> EncoderBlock {
                let attention = MultiHeadSelfAttentionLayer::new(
                    self.sequence_len,
                    self.model_dimension,
                    self.head_count,
                    &self.attention_kqv_weights_init_strategy,
                    &self.attention_dense_layer_init_strategy,
                    &self.rng,
                );
                let network = FeedForwardLayer::new(
                    self.model_dimension,
                    self.feed_forward_hidden_dimension,
                    &self.feed_forward_init_strategy,
                    &self.rng,
                );
                let dropout1 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let dropout2 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let layer_norm1 = LayerNormalization::new(self.sequence_len);
                let layer_norm2 = LayerNormalization::new(self.sequence_len);

                EncoderBlock {
                    attention,
                    network,
                    dropout: (dropout1, dropout2),
                    layer_norm: (layer_norm1, layer_norm2),
                }
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_feed_forward_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.feed_forward_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_hidden_dimension(mut self, hidden_dimension: usize) -> Self {
                self.feed_forward_hidden_dimension = hidden_dimension;
                self
            }

            pub fn with_dropout_rate(mut self, dropout_rate: NodeValue) -> Self {
                self.dropout_rate = dropout_rate;
                self
            }
        }
    }
}

pub mod layers {
    use anyhow::{anyhow, Context, Result};
    use itertools::Itertools;
    use serde::{Deserialize, Serialize};

    use crate::ml::{layer::LayerInitStrategy, NetworkActivationMode, NodeValue, RngStrategy};

    use super::{
        attention::MultiHeadAttention,
        dense::Dense,
        linear::Linear,
        params::{
            keys::{EmbeddingVector, LayerNormalizationBeta, LayerNormalizationGamma},
            TrainableCollection, TrainableLinear, TrainableParameter,
        },
        solver::source::OptimizerSource,
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MultiHeadSelfAttentionLayer {
        attention: MultiHeadAttention,
        dense_layer: Dense,
    }

    impl MultiHeadSelfAttentionLayer {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            kqv_weights_init_strategy: &LayerInitStrategy,
            dense_layer_init_strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let attention = MultiHeadAttention::new(
                sequence_len,
                embedding_dimension,
                head_count,
                kqv_weights_init_strategy,
                rng,
            );

            let dense_layer = Dense::new(
                embedding_dimension,
                embedding_dimension,
                &dense_layer_init_strategy,
                &rng,
            );

            Self {
                attention,
                dense_layer,
            }
        }

        pub fn with_mask(mut self, mask: Linear) -> Self {
            self.attention = self.attention.with_mask(mask);
            self
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            self.forward_with_mask(inputs, None)
        }

        pub fn forward_with_mask(&self, inputs: &Linear, mask: Option<&Linear>) -> Result<Linear> {
            let attention_output = self.attention.forward(inputs, inputs, inputs, mask)?;
            self.dense_layer.forward(&attention_output)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            self.backward_with_mask(inputs, None, output_gradients)
        }

        pub fn backward_with_mask(
            &self,
            inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Linear> {
            // Forward pass
            let attention_output = self.attention.forward(inputs, inputs, inputs, mask)?;

            // Backward pass
            let dense_input_grads = self
                .dense_layer
                .backward(&attention_output, &output_gradients)?;

            let kqv_input_grads =
                self.attention
                    .backward(inputs, inputs, inputs, mask, &dense_input_grads)?;

            self.sum_kqv_gradients(inputs, kqv_input_grads)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.dense_layer
                .apply_gradients(&*optimizer.with_next_index())?;
            self.attention
                .apply_gradients(&*optimizer.with_next_index())?;

            Ok(())
        }

        fn sum_kqv_gradients(
            &self,
            inputs: &Linear,
            kqv_input_grads: Vec<(Linear, Linear, Linear)>,
        ) -> Result<Linear> {
            let zero: Linear = Linear::with_dimensions(inputs);
            let mut sum_inputs_grads = zero.iter().boxed();
            for (k_grads, q_grads, v_grads) in &kqv_input_grads {
                let sum = Linear::sum([k_grads, q_grads, v_grads].into_iter()).unwrap();
                sum_inputs_grads = sum_inputs_grads.add(sum).boxed();
            }
            Ok(sum_inputs_grads.collect())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MultiHeadCrossAttentionLayer {
        attention: MultiHeadAttention,
        dense_layer: Dense,
    }

    impl MultiHeadCrossAttentionLayer {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            kqv_weights_init_strategy: &LayerInitStrategy,
            dense_layer_init_strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let attention = MultiHeadAttention::new(
                sequence_len,
                embedding_dimension,
                head_count,
                kqv_weights_init_strategy,
                rng,
            );

            let dense_layer = Dense::new(
                embedding_dimension,
                embedding_dimension,
                &dense_layer_init_strategy,
                &rng,
            );

            Self {
                attention,
                dense_layer,
            }
        }

        pub fn forward(&self, kv_inputs: &Linear, q_inputs: &Linear) -> Result<Linear> {
            self.forward_with_mask(kv_inputs, q_inputs, None)
        }

        pub fn forward_with_mask(
            &self,
            kv_inputs: &Linear,
            q_inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let attention_output = self
                .attention
                .forward(kv_inputs, q_inputs, kv_inputs, mask)?;
            self.dense_layer.forward(&attention_output)
        }

        pub fn backward(
            &mut self,
            kv_inputs: &Linear,
            q_inputs: &Linear,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear)> {
            self.backward_with_mask(kv_inputs, q_inputs, None, output_gradients)
        }

        pub fn backward_with_mask(
            &self,
            kv_inputs: &Linear,
            q_inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear)> {
            // Forward pass
            let attention_output = self
                .attention
                .forward(kv_inputs, q_inputs, kv_inputs, mask)?;

            // Backward pass
            let dense_input_grads = self
                .dense_layer
                .backward(&attention_output, &output_gradients)?;

            let kqv_input_grads = self.attention.backward(
                kv_inputs,
                q_inputs,
                kv_inputs,
                mask,
                &dense_input_grads,
            )?;

            self.sum_kv_q_gradients(kv_inputs, kqv_input_grads)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.dense_layer
                .apply_gradients(&*optimizer.with_next_index())?;
            self.attention
                .apply_gradients(&*optimizer.with_next_index())?;

            Ok(())
        }

        fn sum_kv_q_gradients(
            &self,
            inputs: &Linear,
            kqv_input_grads: Vec<(Linear, Linear, Linear)>,
        ) -> Result<(Linear, Linear)> {
            let zero: Linear = Linear::with_dimensions(inputs);
            let mut sum_kv_inputs_grads = zero.iter().boxed();
            let mut sum_q_inputs_grads = zero.iter().boxed();

            for (k_grads, q_grads, v_grads) in &kqv_input_grads {
                let sum_kv = Linear::sum([k_grads, v_grads].into_iter()).unwrap();
                sum_kv_inputs_grads = sum_kv_inputs_grads.add(sum_kv).boxed();
                sum_q_inputs_grads = sum_q_inputs_grads.add(q_grads.iter()).boxed();
            }

            let kv_inputs_grads = sum_kv_inputs_grads.collect();
            let q_inputs_grads = sum_q_inputs_grads.collect();

            Ok((kv_inputs_grads, q_inputs_grads))
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FeedForwardLayer {
        hidden_layer: Dense,
        output_layer: Dense,
    }

    impl FeedForwardLayer {
        pub fn new(
            model_dimensions: usize,
            hidden_dimension: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut hidden_layer = Dense::new(model_dimensions, hidden_dimension, strategy, rng);
            hidden_layer.set_activation(NetworkActivationMode::RelU);

            let output_layer = Dense::new(hidden_dimension, model_dimensions, strategy, rng);
            Self {
                hidden_layer,
                output_layer,
            }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            let hidden_layer = self.hidden_layer.forward(&inputs)?;
            let layer_output = self.output_layer.forward(&hidden_layer)?;

            Ok(layer_output)
        }

        pub fn backward(&self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let final_layer_inputs = self.hidden_layer.forward(&inputs)?;
            let final_layer_gradients = self
                .output_layer
                .backward(&final_layer_inputs, &output_gradients)?;

            let dense_input_gradients = self
                .hidden_layer
                .backward(&inputs, &final_layer_gradients)?;

            Ok(dense_input_gradients)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.hidden_layer
                .apply_gradients(&*optimizer.with_next_index())?;
            self.output_layer
                .apply_gradients(&*optimizer.with_next_index())?;

            Ok(())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingLayer {
        embeddings: Vec<TrainableLinear>,
        vocab_size: usize,
        model_dimensions: usize,
    }

    impl TrainableCollection<EmbeddingVector> for EmbeddingLayer {
        fn store(&self, index: usize) -> Option<super::params::ParameterStore> {
            Some(self.embeddings.get(index)?.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut [TrainableLinear]> {
            Some(&mut self.embeddings)
        }
    }

    impl EmbeddingLayer {
        pub fn new(
            model_dimensions: usize,
            vocab_size: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut linear = Linear::new(vocab_size, model_dimensions);
            linear.initialize_as_layer(strategy, rng);
            let embeddings = linear
                .to_values()
                .into_iter()
                .map(|row| Linear::from_values(&[row]).unwrap().into())
                .collect();

            Self {
                embeddings,
                vocab_size,
                model_dimensions,
            }
        }

        pub fn forward<T: AsRef<[usize]>>(&self, token_sequence: T) -> Result<Linear> {
            let mut embedding_vectors = vec![];

            for &token in token_sequence.as_ref() {
                if token >= self.vocab_size {
                    return Err(anyhow!("token not in embedding range"));
                }
                let embedding = self.embeddings.get(token).context("invalid token id")?;
                let embedding_vector = embedding.value().as_single_stride()?;
                embedding_vectors.push(embedding_vector);
            }

            Linear::from_values(&embedding_vectors)
        }

        pub fn backward<T: AsRef<[usize]>>(
            &self,
            token_sequence: T,
            output_gradients: &Linear,
        ) -> Result<()> {
            for (&token, grad) in token_sequence
                .as_ref()
                .iter()
                .zip(output_gradients.rows_iter())
            {
                let grad = Linear::from_iter(grad.len(), grad.iter().copied()).unwrap();
                self.queue_gradients(EmbeddingVector, token, grad.iter().boxed());
            }

            Ok(())
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.apply_param_gradients(EmbeddingVector, optimizer)?;
            Ok(())
        }

        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        pub fn pad_inputs(
            input_sequence: &[usize],
            seq_len: usize,
            pad_token: usize,
        ) -> (Vec<usize>, Linear) {
            let input_token_len = input_sequence.len();

            let mut input_sequence = input_sequence.to_vec();
            let mask = if input_token_len == seq_len {
                Linear::with_value(seq_len, 1, 1.0)
            } else {
                input_sequence.resize(seq_len, pad_token);
                let mask_iter = (0..seq_len).map(|i| if i < input_token_len { 1.0 } else { 0.0 });
                Linear::from_iter(1, mask_iter).unwrap()
            };

            (input_sequence, mask)
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PositionalEmbeddingLayer {
        embeddings: EmbeddingLayer,
    }

    impl PositionalEmbeddingLayer {
        pub fn new(
            model_dimensions: usize,
            max_sequence_len: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            Self {
                embeddings: EmbeddingLayer::new(model_dimensions, max_sequence_len, strategy, &rng),
            }
        }

        pub fn forward(&self, start_index: usize, count: usize) -> Result<Linear> {
            let end_index = start_index + count;
            if end_index > self.embeddings.vocab_size() {
                return Err(anyhow!("positional index out of range"))?;
            }

            let sequence = start_index..end_index;
            self.embeddings.forward(&sequence.collect_vec())
        }

        pub fn backward(
            &self,
            start_index: usize,
            count: usize,
            output_gradients: &Linear,
        ) -> Result<()> {
            let end_index = start_index + count;
            if end_index > self.embeddings.vocab_size() {
                return Err(anyhow!("positional index out of range"))?;
            }
            if output_gradients.count() != count {
                return Err(anyhow!("mismatched gradient vector count"))?;
            }

            let sequence = start_index..end_index;
            self.embeddings
                .backward(&sequence.collect_vec(), output_gradients)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.embeddings.apply_gradients(optimizer)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DropoutLayer {
        dropout_rate: NodeValue,
        #[serde(default)]
        rng: RngStrategy,
    }

    impl DropoutLayer {
        pub fn new(dropout_rate: NodeValue, rng: &RngStrategy) -> Self {
            let rng = rng.clone().upgrade();
            Self { dropout_rate, rng }
        }

        pub fn forward(&self, input: &Linear) -> Result<Linear> {
            Ok(self.forward_advanced(&input)?.0)
        }

        pub fn forward_advanced(&self, input: &Linear) -> Result<(Linear, Linear)> {
            let rng = RngStrategy::default();
            let dropout_mask = input.iter().dropout_mask(self.dropout_rate, rng);
            let masked_output = input.iter().dot_product(dropout_mask.iter()).collect();
            Ok((masked_output, dropout_mask))
        }

        pub fn forward_if_enabled(&self, input: Linear, enabled: bool) -> Result<Linear> {
            if enabled {
                Ok(self.forward_advanced(&input)?.0)
            } else {
                Ok(input)
            }
        }

        pub fn backward(
            &mut self,
            dropout_mask: &Linear,
            output_gradients: &Linear,
        ) -> Result<Linear> {
            let grad_input = output_gradients
                .iter()
                .dot_product(dropout_mask.iter())
                .collect();
            Ok(grad_input)
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LayerNormalization {
        beta: TrainableLinear,
        gamma: TrainableLinear,
    }

    impl LayerNormalization {
        pub fn new(seq_len: usize) -> Self {
            Self {
                beta: Linear::with_value(seq_len, 1, 0.0).into(),
                gamma: Linear::with_value(seq_len, 1, 1.0).into(),
            }
        }

        pub fn forward(&self, input: &Linear) -> Result<Linear> {
            let stride = input.stride();
            // mean(x) = sum(x) / N
            let mean = input.iter().flatten_mean();
            // std(x) = sum((x - mean(x))^2)^0.5 * (N - 1)^-0.5
            let std_dev = input.iter().flatten_stddev(mean.iter(), false);

            // norm(x) = x - mean(x) / std(x)
            let normalised_input = input
                .iter()
                .sub(mean.iter().grow(stride))
                .div(std_dev.iter().grow(stride), Some(1e-8))
                .collect();

            // layer_norm(x) = norm(x) * gamma + beta
            let norm_scaled_shifted = normalised_input
                .iter()
                .dot_product(self.gamma.iter().grow(stride))
                .add(self.beta.iter().grow(stride))
                .collect();

            Ok(norm_scaled_shifted)
        }

        pub fn backward(&self, x: &Linear, dl_dnorm: &Linear) -> Result<Linear> {
            let row_x = x.rows_iter().map(|row| Linear::from_values(&[row.into()]));
            let rows_grads = dl_dnorm
                .rows_iter()
                .map(|row| Linear::from_values(&[row.into()]));
            let gamma = self.gamma.value().values_iter().map(|(&g, ..)| g);

            let mut dloss_dx = Linear::with_dimensions(x);
            let mut dloss_dbeta = Linear::with_dimensions(self.beta.value());
            let mut dloss_dgamma = Linear::with_dimensions(self.gamma.value());

            for (i, ((row, grad), gamma)) in row_x.zip(rows_grads).zip(gamma).enumerate() {
                let (dx, dbeta, dgamma) = self.backward_jacobian(&row?, gamma, &grad?)?;
                dloss_dx.copy_stride_into(&dx, 0, i);
                dloss_dbeta.copy_stride_into(&dbeta, 0, i);
                dloss_dgamma.copy_stride_into(&dgamma, 0, i);
            }

            self.queue_gradients(LayerNormalizationBeta, dloss_dbeta.iter().boxed());
            self.queue_gradients(LayerNormalizationGamma, dloss_dgamma.iter().boxed());

            Ok(dloss_dx)
        }

        pub fn backward_jacobian(
            &self,
            x: &Linear,
            gamma: NodeValue,
            dl_dnorm: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            // Compute mean and standard deviation of input
            let mean = x.iter().flatten_mean();
            let std_dev = x.iter().flatten_stddev(mean.iter(), false);

            let stride = x.stride();
            let batch_size = x.stride() as NodeValue;

            let mean_scalar = mean.as_scalar().unwrap();
            let std_dev_scalar = std_dev.as_scalar().unwrap();
            let epsilon = 1e-8;

            let x_minus_mean = x.iter().sub_scalar(mean_scalar).collect();
            let norm_input = x_minus_mean
                .iter()
                .multiply_scalar((std_dev_scalar + epsilon).powi(-1));

            let dbeta = dl_dnorm.iter().flatten_sum();
            let dgamma = dl_dnorm.iter().dot_product(norm_input).flatten_sum();

            // dμ = ones(n, n) .* 1/n
            let dmean = 1.0 / batch_size;

            // dσ = (x .- means[k]) ./ (n * stds[k])
            let dstd = x_minus_mean
                .iter_transpose()
                .multiply_scalar((batch_size * std_dev_scalar).powi(-1)); // [1, N]

            // dx = (I(n) - dμ) ./ (stds[k] + model.ϵ) - dσ * transpose(x .- means[k]) ./ (stds[k] + model.ϵ).^2
            let factor = gamma / (std_dev_scalar + epsilon);
            let dx = Linear::diagonal_iter(stride, factor)
                .sub_scalar(dmean * factor)
                .sub(
                    dstd.grow(stride)
                        .dot_product(x_minus_mean.iter().stack(stride))
                        .multiply_scalar(gamma / ((std_dev_scalar * std_dev_scalar) + epsilon)),
                )
                .collect(); // [N, N]

            // dL/dx = dz/dx * dL/dz  =>  dloss_dx.T = dx * dl_dnorm.T  =>  dl_dnorm * dx.T = dloss_dx
            let dloss_dx = dl_dnorm.matrix_product_rhs_transposed(&dx);
            Ok((dloss_dx, dbeta, dgamma))
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.apply_param_gradients(LayerNormalizationBeta, optimizer)?;
            self.apply_param_gradients(LayerNormalizationGamma, optimizer)?;
            Ok(())
        }
    }

    impl TrainableParameter<LayerNormalizationBeta> for LayerNormalization {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.beta.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.beta)
        }
    }

    impl TrainableParameter<LayerNormalizationGamma> for LayerNormalization {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.gamma.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.gamma)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::{
                solver,
                tests::helpers::{
                    assert_input_gradients, assert_optimisation_converges, new_linear,
                },
            },
            RngStrategy,
        };

        use super::*;

        #[test]
        fn feed_forward_layer_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let hidden_dim = 48;

            let rng = RngStrategy::testable(1234);
            let strategy = LayerInitStrategy::KaimingZeroBias;

            let feed_forward_layer = FeedForwardLayer::new(embed_dim, hidden_dim, &strategy, &rng);

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = feed_forward_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn layer_normalization_layer_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let layer_norm = LayerNormalization::new(seq_len);

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = layer_norm.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn feed_forward_layer_can_compute_valid_gradients() {
            let seq_len = 3;
            let embed_dim = 12;
            let hidden_dim = 48;
            let strategy = LayerInitStrategy::KaimingZeroBias;

            assert_input_gradients(
                &move |rng| {
                    let ff_layer = FeedForwardLayer::new(embed_dim, hidden_dim, &strategy, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (ff_layer, inputs, target)
                },
                &move |ff_layer, inputs| ff_layer.forward(&inputs).unwrap(),
                &move |ff_layer, inputs, dloss| ff_layer.backward(&inputs, &dloss).unwrap(),
            );
        }

        #[test]
        fn layer_normalization_layer_can_compute_valid_gradients() {
            let seq_len = 2;
            let embed_dim = 4;

            assert_input_gradients(
                &move |rng| {
                    let layer_norm = LayerNormalization::new(seq_len);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (layer_norm, inputs, target)
                },
                &move |layer_norm, inputs| layer_norm.forward(&inputs).unwrap(),
                &move |layer_norm, inputs, dloss| layer_norm.backward(&inputs, &dloss).unwrap(),
            );
        }

        #[test]
        fn feed_forward_layer_can_optimise() {
            let seq_len = 3;
            let embed_dim = 12;
            let hidden_dim = 48;
            let strategy = LayerInitStrategy::KaimingZeroBias;

            let learn_rate = 0.01;
            let iters = 15;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let ff_layer = FeedForwardLayer::new(embed_dim, hidden_dim, &strategy, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (ff_layer, inputs, target)
                },
                &move |ff_layer, inputs| ff_layer.forward(&inputs).unwrap(),
                &move |ff_layer, inputs, dloss| {
                    let grads = ff_layer.backward(&inputs, &dloss).unwrap();
                    ff_layer.apply_gradients(&optimizer).unwrap();
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
        }

        #[test]
        fn layer_normalization_layer_can_optimise_inputs() {
            let seq_len = 3;
            let embed_dim = 12;

            let learn_rate = 0.001;
            let iters = 500;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let layer_norm = LayerNormalization::new(seq_len);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (layer_norm, inputs, target)
                },
                &move |layer_norm, inputs| layer_norm.forward(&inputs).unwrap(),
                &move |layer_norm, inputs, dloss| {
                    let grads = layer_norm.backward(&inputs, &dloss).unwrap();
                    layer_norm.apply_gradients(&optimizer).unwrap();
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
        }
    }
}

pub mod attention {
    use std::iter;

    use anyhow::{anyhow, Result};
    use serde::{Deserialize, Serialize};

    use crate::ml::{layer::LayerInitStrategy, NetworkActivationMode, NodeValue, RngStrategy};

    use super::{
        linear::Linear,
        params::{
            keys::{AttentionKeyWeights, AttentionQueryWeights, AttentionValueWeights},
            TrainableLinear, TrainableParameter,
        },
        solver::source::OptimizerSource,
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MultiHeadAttention {
        heads: Vec<AttentionHead>,
    }

    impl MultiHeadAttention {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            kqv_weights_init_strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let heads = iter::repeat_with(|| {
                AttentionHead::new_head(
                    sequence_len,
                    embedding_dimension,
                    head_count,
                    kqv_weights_init_strategy,
                    rng,
                )
            })
            .take(head_count)
            .collect();

            Self { heads }
        }

        pub fn with_mask(mut self, mask: Linear) -> Self {
            self.heads
                .iter_mut()
                .for_each(|head| head.set_mask(Some(mask.clone())));
            self
        }

        pub fn forward(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let mut head_outputs = vec![];
            for head in &self.heads {
                let head_output =
                    head.forward_advanced(key_inputs, query_inputs, value_inputs, mask)?;
                head_outputs.push(head_output);
            }

            let attention_output = head_outputs
                .into_iter()
                .fold(None, |concat: Option<Linear>, head| match concat {
                    Some(iter) => Some(iter.concat(&head).collect()),
                    None => Some(head),
                })
                .unwrap();

            Ok(attention_output)
        }

        pub fn backward(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Vec<(Linear, Linear, Linear)>> {
            let head_output_gradients = output_gradients.split(self.heads.len());

            let mut kqv_input_grads = vec![];

            for (head, output_gradients) in self.heads.iter().zip(&head_output_gradients) {
                let head_input_gradients = head.backward_advanced(
                    key_inputs,
                    query_inputs,
                    value_inputs,
                    mask,
                    &output_gradients,
                )?;
                kqv_input_grads.push(head_input_gradients);
            }

            Ok(kqv_input_grads)
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            for head in self.heads.iter_mut() {
                head.apply_gradients(optimizer)?;
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AttentionHead {
        key_weights: TrainableLinear,
        query_weights: TrainableLinear,
        value_weights: TrainableLinear,
        mask: Option<Linear>,
        embedding_dimension: usize,
        sequence_len: usize,
    }

    impl AttentionHead {
        pub fn new(sequence_len: usize, embedding_dimension: usize, rng: &RngStrategy) -> Self {
            Self::new_head(
                sequence_len,
                embedding_dimension,
                1,
                &LayerInitStrategy::FullRandom,
                rng,
            )
        }

        pub fn new_head(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            kqv_weights_init_strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let strategy = kqv_weights_init_strategy;
            let head_dim = embedding_dimension / head_count;
            Self {
                key_weights: Self::new_kqv_linear(embedding_dimension, head_dim, strategy, rng),
                query_weights: Self::new_kqv_linear(embedding_dimension, head_dim, strategy, rng),
                value_weights: Self::new_kqv_linear(embedding_dimension, head_dim, strategy, rng),
                mask: None,
                embedding_dimension,
                sequence_len,
            }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            if inputs.count() == 0 {
                Err(anyhow!("no values provided for input vectors"))?;
            }
            if inputs.count() != self.sequence_len {
                Err(anyhow!("mismatched number of input vectors"))?;
            }
            if inputs.stride() != self.embedding_dimension {
                Err(anyhow!("mismatched input vector size"))?;
            };

            let mask = self.mask.as_ref();
            self.forward_advanced(&inputs, &inputs, &inputs, mask)
        }

        pub fn forward_advanced(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let keys = key_inputs.matrix_product(self.key_weights.value());
            let queries = query_inputs.matrix_product(self.query_weights.value());
            let values = value_inputs.matrix_product(self.value_weights.value());

            let output = Self::scaled_attention(&keys, &queries, &values, mask, self.mask.as_ref());
            Ok(output)
        }

        pub fn backward(&self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let kqv_grads =
                self.backward_advanced(&inputs, &inputs, &inputs, None, output_gradients)?;

            let (k_grads, q_grads, v_grads) = kqv_grads;
            let summed_grads = k_grads.iter().add(q_grads.iter()).add(v_grads.iter());
            let grads = summed_grads.collect();
            Ok(grads)
        }

        pub fn backward_advanced(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            if output_gradients.stride() != self.value_weights.stride() {
                Err(anyhow!("mismatched ouput vector size"))?;
            }

            let keys = key_inputs.matrix_product(self.key_weights.value());
            let queries = query_inputs.matrix_product(self.query_weights.value());
            let values = value_inputs.matrix_product(self.value_weights.value());

            let (dkeys, dqueries, dvalues) = Self::scaled_attention_d(
                &keys,
                &queries,
                &values,
                mask,
                self.mask.as_ref(),
                &output_gradients,
            )?;

            let dkey_weights = key_inputs.matrix_product_lhs_transposed(&dkeys);
            let dquery_weights = query_inputs.matrix_product_lhs_transposed(&dqueries);
            let dvalue_weights = value_inputs.matrix_product_lhs_transposed(&dvalues);

            self.queue_gradients(AttentionKeyWeights, dkey_weights.iter().boxed());
            self.queue_gradients(AttentionQueryWeights, dquery_weights.iter().boxed());
            self.queue_gradients(AttentionValueWeights, dvalue_weights.iter().boxed());

            let dkey_inputs = dkeys.matrix_product_rhs_transposed(self.key_weights.value());
            let dquery_inputs = dqueries.matrix_product_rhs_transposed(self.query_weights.value());
            let dvalue_inputs = dvalues.matrix_product_rhs_transposed(self.value_weights.value());

            Ok((dkey_inputs, dvalue_inputs, dquery_inputs))
        }

        fn scaled_attention(
            keys: &Linear,
            queries: &Linear,
            values: &Linear,
            input_mask: Option<&Linear>,
            mask: Option<&Linear>,
        ) -> Linear {
            // attention_scores = queries * keys.T
            let attention_scores = queries.matrix_product_rhs_transposed(&keys);
            let scale_factor = (keys.count() as NodeValue).powf(-0.5);
            let scaled_attention_scores = attention_scores.iter().multiply_scalar(scale_factor);

            let masked_attention_scores = match mask {
                Some(mask) => scaled_attention_scores
                    .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                    .boxed(),
                None => scaled_attention_scores.boxed(),
            };

            let stride = attention_scores.stride();
            let input_mask = input_mask.map(|mask| mask.iter().grow(stride));
            let attention_weights = match input_mask {
                Some(mask) => masked_attention_scores
                    .set_mask(mask, NodeValue::NEG_INFINITY)
                    .softmax(),
                None => masked_attention_scores.softmax(),
            };

            attention_weights.matrix_product(&values)
        }

        fn scaled_attention_d(
            keys: &Linear,
            queries: &Linear,
            values: &Linear,
            input_mask: Option<&Linear>,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            // forward pass
            let attention_scores = queries.matrix_product_rhs_transposed(&keys);
            let scale_factor = (keys.count() as NodeValue).powf(-0.5);
            let scaled_attention_scores = attention_scores.iter().multiply_scalar(scale_factor);

            let masked_attention_scores = match mask {
                Some(mask) => scaled_attention_scores
                    .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                    .boxed(),
                None => scaled_attention_scores.boxed(),
            };

            let stride = attention_scores.stride();
            let input_mask = input_mask.map(|mask| mask.iter().grow(stride).collect());
            let attention_weights = match &input_mask {
                Some(mask) => masked_attention_scores
                    .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                    .softmax(),
                None => masked_attention_scores.softmax(),
            };

            // backward pass
            let dvalues = attention_weights.matrix_product_lhs_transposed(&output_gradients);
            let dattention_weights = output_gradients.matrix_product_rhs_transposed(&values);
            let softmax_grads = softmax_d_iter(dattention_weights, attention_weights)?;
            // let dmasked_attention_scores = dattention_weights.softmax_d(&attention_weights);

            let dmasked_attention_scores = match &input_mask {
                Some(mask) => softmax_grads.iter().set_mask(mask.iter(), 0.0).boxed(),
                None => softmax_grads.iter().boxed(),
            };

            let dscaled_attention_scores = match mask {
                Some(mask) => dmasked_attention_scores.set_mask(mask.iter(), 0.0).boxed(),
                None => dmasked_attention_scores.boxed(),
            };

            let dattention_scores = dscaled_attention_scores
                .multiply_scalar(scale_factor)
                .collect();

            // attention_scores = queries * keys.T
            // dattention_scores = dqueries * keys.T
            // dattention_scores * keys = dqueries
            let dqueries = dattention_scores.matrix_product(&keys);
            // attention_scores = queries * keys.T
            // dattention_scores = queries * dkeys.T
            // dkeys = dattention_scores.T * queries
            let dkeys = dattention_scores.matrix_product_lhs_transposed(&queries);

            Ok((dkeys, dqueries, dvalues))
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.apply_param_gradients(AttentionKeyWeights, optimizer)?;
            self.apply_param_gradients(AttentionQueryWeights, optimizer)?;
            self.apply_param_gradients(AttentionValueWeights, optimizer)?;
            Ok(())
        }

        fn new_kqv_linear(
            embedding_dimension: usize,
            head_dimension: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> TrainableLinear {
            let mut linear = Linear::new(embedding_dimension, head_dimension);
            linear.initialize_as_layer(&strategy, &rng);
            linear.into()
        }

        pub fn set_mask(&mut self, mask: Option<Linear>) {
            self.mask = mask;
        }
    }

    // TODO: benchmark against LinearIter::softmax_d, delete loser!
    fn softmax_d_iter(dloss_dsoftmax: Linear, softmax: Linear) -> Result<Linear> {
        // let attention_weights = masked_attention_scores.softmax();
        Linear::from_iter(
            softmax.stride(),
            dloss_dsoftmax
                .rows_iter()
                .zip(softmax.rows_iter())
                .flat_map(|(grads, probs)| NetworkActivationMode::softmax_d(grads, probs)),
        )
    }

    impl TrainableParameter<AttentionKeyWeights> for AttentionHead {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.key_weights.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.key_weights)
        }
    }

    impl TrainableParameter<AttentionQueryWeights> for AttentionHead {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.query_weights.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.query_weights)
        }
    }

    impl TrainableParameter<AttentionValueWeights> for AttentionHead {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.value_weights.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.value_weights)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::transformer::{
            layers::MultiHeadSelfAttentionLayer,
            solver,
            tests::helpers::{assert_input_gradients, assert_optimisation_converges, new_linear},
        };

        use super::*;

        #[test]
        fn attention_can_compute_outputs_single_head() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let attention_layer = AttentionHead::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn attention_can_compute_outputs_multi_head_layer() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);
            let strategy = LayerInitStrategy::KaimingZeroBias;

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let attention_layer =
                MultiHeadSelfAttentionLayer::new(seq_len, embed_dim, 3, &strategy, &strategy, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn attention_can_compute_for_kvq() {
            let seq_len = 3;
            let embed_dim = 4;

            let rng = RngStrategy::testable(1234);

            let queries = new_linear(seq_len, embed_dim, &rng);
            let keys = new_linear(seq_len, embed_dim, &rng);
            let values = new_linear(seq_len, embed_dim, &rng);

            let output = AttentionHead::scaled_attention(&queries, &keys, &values, None, None);

            assert_eq!(output.count(), seq_len);
            assert_eq!(output.stride(), embed_dim);
        }

        #[test]
        fn attention_can_compute_gradients_single_head() {
            let seq_len = 5;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_linear(seq_len, embed_dim, &rng);

            let attention_layer = AttentionHead::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            let output_gradients = &new_linear(seq_len, embed_dim, &rng);
            let output_gradients = output_gradients.iter().multiply_scalar(0.01).collect();

            let grads = attention_layer
                .backward(&inputs, &output_gradients)
                .unwrap();

            let grads = grads.to_values();

            assert_eq!(grads.len(), output.len());
            assert_eq!(grads[0].len(), output[0].len());
        }

        #[test]
        fn attention_can_compute_valid_gradients_for_single_head() {
            let seq_len = 5;
            let embed_dim = 12;

            assert_input_gradients(
                &move |rng| {
                    let attention_layer = AttentionHead::new(seq_len, embed_dim, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_layer, inputs, target)
                },
                &move |attention_layer, inputs| attention_layer.forward(&inputs).unwrap(),
                &move |attention_layer, inputs, dloss| {
                    attention_layer.backward(&inputs, &dloss).unwrap()
                },
            );
        }

        #[test]
        fn attention_can_compute_valid_gradients_for_multi_head_layer() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;

            assert_input_gradients(
                &move |rng| {
                    let attention_layer = MultiHeadSelfAttentionLayer::new(
                        seq_len,
                        embed_dim,
                        head_count,
                        &LayerInitStrategy::KaimingZeroBias,
                        &LayerInitStrategy::KaimingZeroBias,
                        &rng,
                    );
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_layer, inputs, target)
                },
                &move |attention_layer, inputs| attention_layer.forward(&inputs).unwrap(),
                &move |attention_layer, inputs, dloss| {
                    attention_layer.backward(&inputs, &dloss).unwrap()
                },
            );
        }

        #[test]
        fn attention_can_minimise_single_head() {
            let seq_len = 3;
            let embed_dim = 12;
            let learn_rate = 0.001;
            let total_iterations = 25;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let attention_head = AttentionHead::new(seq_len, embed_dim, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_head, inputs, target)
                },
                &move |attention_head, inputs| attention_head.forward(&inputs).unwrap(),
                &move |attention_head, inputs, dloss| {
                    let grads = attention_head.backward(&inputs, &dloss).unwrap();

                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);
                    attention_head.apply_gradients(&optimizer).unwrap();
                    grads
                },
                total_iterations,
            );
        }

        #[test]
        fn attention_can_minimise_multi_head_layer() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let learn_rate = 0.001;
            let total_iterations = 25;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let attention_layer = MultiHeadSelfAttentionLayer::new(
                        seq_len,
                        embed_dim,
                        head_count,
                        &LayerInitStrategy::KaimingZeroBias,
                        &LayerInitStrategy::KaimingZeroBias,
                        &rng,
                    );
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_layer, inputs, target)
                },
                &move |attention_layer, inputs| attention_layer.forward(&inputs).unwrap(),
                &move |attention_layer, inputs, dloss| {
                    let grads = attention_layer.backward(&inputs, &dloss).unwrap();

                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);
                    attention_layer.apply_gradients(&optimizer).unwrap();
                    grads
                },
                total_iterations,
            );
        }
    }
}

pub mod dense {
    use anyhow::{anyhow, Result};
    use serde::{Deserialize, Serialize};

    use crate::ml::{layer::LayerInitStrategy, LayerValues, NetworkActivationMode, RngStrategy};

    use super::{
        linear::Linear,
        params::{
            keys::{DenseBias, DenseWeight},
            TrainableLinear, TrainableParameter,
        },
        solver::source::OptimizerSource,
    };

    #[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
    pub struct Dense {
        weights: TrainableLinear,
        bias: Option<TrainableLinear>,
        activation: Option<NetworkActivationMode>,
        inputs_count: usize,
        outputs_count: usize,
    }

    impl Dense {
        pub fn new(
            inputs_count: usize,
            outputs_count: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut weights = Linear::new(inputs_count, outputs_count);
            weights.initialize_as_layer(strategy, rng);
            let weights = weights.into();

            let bias = if strategy.requires_bias() {
                let mut b = Linear::new(1, outputs_count);
                b.initialize_as_layer_bias(strategy, rng);
                Some(b.into())
            } else {
                None
            };

            Self {
                weights,
                bias,
                activation: None,
                inputs_count,
                outputs_count,
            }
        }

        pub fn forward_row(&self, inputs: LayerValues) -> Result<LayerValues> {
            if inputs.len() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }

            let inputs = Linear::from_values(&[inputs])?;
            let outputs = self.forward(&inputs)?;
            let row = outputs.as_single_stride()?;

            Ok(row)
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            if inputs.stride() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }
            let mut weighted_inputs = self.compute_weighted_inputs(&inputs)?;

            let outputs = match &self.activation {
                Some(activation) => {
                    weighted_inputs.rows_iter_mut().for_each(|row| {
                        let activation_row = activation.apply(&row.into());
                        row.clone_from_slice(&activation_row)
                    });
                    weighted_inputs
                }
                None => weighted_inputs,
            };

            Ok(outputs)
        }

        fn compute_weighted_inputs(&self, inputs: &Linear) -> Result<Linear> {
            // -> output = inputs * weights
            // TODO: (PERF) optimize matmul operation by ensuring matrix dims are aligned to 16 elems
            let output = inputs.matrix_product(self.weights.value());

            if let Some(bias) = &self.bias {
                let bias = bias.iter().stack(inputs.count());
                Ok(output.iter().add(bias).collect())
            } else {
                Ok(output)
            }
        }

        pub fn backward(&self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let weighted_inputs_gradients = match &self.activation {
                Some(activation) => {
                    let mut weighted_inputs = self.compute_weighted_inputs(&inputs)?;
                    weighted_inputs
                        .rows_iter_mut()
                        .zip(output_gradients.rows_iter())
                        .for_each(|(x, grads)| {
                            let output_gradients = grads.into();
                            let outputs = x.into();
                            let activated_inputs = activation.apply(&outputs);
                            activation
                                .derivative(&output_gradients, &activated_inputs)
                                .multiply_iter(&output_gradients)
                                .zip(x)
                                .for_each(|(grad, x)| *x = grad)
                        });
                    weighted_inputs
                }
                None => output_gradients.clone(),
            };

            let bias_gradients = weighted_inputs_gradients.iter_transpose().flatten_sum();

            // -> output = inputs * weights
            // -> weighted_inputs_gradients = inputs * weights_gradients
            // -> weights_gradients = inputs.T * weighted_inputs_gradients
            let weights_gradients =
                inputs.matrix_product_lhs_transposed(&weighted_inputs_gradients);

            self.queue_gradients(DenseWeight, weights_gradients.iter().boxed());
            if self.bias.is_some() {
                self.queue_gradients(DenseBias, bias_gradients.iter_transpose().boxed());
            }

            // -> output = inputs * weights
            // -> weighted_inputs_gradients = input_gradients_m * self.weights
            // -> input_gradients_m = weighted_inputs_gradients * self.weights.T
            let input_gradients =
                weighted_inputs_gradients.matrix_product_rhs_transposed(self.weights.value());

            Ok(input_gradients)
        }

        pub fn backward_row(
            &self,
            inputs: LayerValues,
            output_gradients: LayerValues,
        ) -> Result<LayerValues> {
            let inputs = Linear::from_values(&[inputs])?;
            let output_gradients = Linear::from_values(&[output_gradients])?;
            let input_gradients_m = self.backward(&inputs, &output_gradients)?;

            input_gradients_m.as_single_stride()
        }

        pub fn set_activation(&mut self, activation: NetworkActivationMode) {
            self.activation = Some(activation);
        }

        pub fn apply_gradients<T: OptimizerSource>(&mut self, optimizer: &T) -> Result<()> {
            self.apply_param_gradients(DenseWeight, optimizer)?;
            if self.bias.is_some() {
                self.apply_param_gradients(DenseBias, optimizer)?;
            }
            Ok(())
        }
    }

    impl TrainableParameter<DenseWeight> for Dense {
        fn store(&self) -> Option<super::params::ParameterStore> {
            Some(self.weights.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            Some(&mut self.weights)
        }
    }

    impl TrainableParameter<DenseBias> for Dense {
        fn store(&self) -> Option<super::params::ParameterStore> {
            self.bias.as_ref().map(|x| x.parameters())
        }

        fn param_mut(&mut self) -> Option<&mut TrainableLinear> {
            self.bias.as_mut()
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::transformer::{
            solver,
            tests::helpers::{assert_input_gradients, assert_optimisation_converges, new_linear},
        };

        use super::*;

        #[test]
        fn dense_can_minimise() {
            let seq_len = 4;
            let embed_dim = 8;
            let output_dim = 12;
            let strategy = LayerInitStrategy::Kaiming;

            let learn_rate = 0.01;
            let total_iterations = 25;
            let optimizer = solver::SGDOptimizer::new_cache(learn_rate);

            assert_optimisation_converges(
                &move |rng| {
                    let dense = Dense::new(embed_dim, output_dim, &strategy, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, output_dim, &rng);
                    (dense, inputs, target)
                },
                &move |dense, inputs| dense.forward(&inputs).unwrap(),
                &move |dense, inputs, dloss| {
                    let grads = dense.backward(&inputs, &dloss).unwrap();

                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);
                    dense.apply_gradients(&optimizer).unwrap();
                    grads
                },
                total_iterations,
            );
        }

        #[test]
        fn dense_can_compute_valid_gradients_for_simple_feed_forward() {
            computed_dloss_dinput_delta(|_| (), LayerInitStrategy::Kaiming);
        }

        #[test]
        fn dense_can_compute_valid_gradients_for_relu_feed_forward() {
            computed_dloss_dinput_delta(
                |dense| dense.set_activation(NetworkActivationMode::RelU),
                LayerInitStrategy::Kaiming,
            );
        }

        #[test]
        fn dense_can_compute_valid_gradients_for_tanh_feed_forward() {
            computed_dloss_dinput_delta(
                |dense| dense.set_activation(NetworkActivationMode::Tanh),
                LayerInitStrategy::Kaiming,
            );
        }

        #[test]
        fn dense_can_compute_valid_gradients_for_sigmoid_feed_forward() {
            computed_dloss_dinput_delta(
                |dense| dense.set_activation(NetworkActivationMode::Sigmoid),
                LayerInitStrategy::Kaiming,
            );
        }

        fn computed_dloss_dinput_delta(
            configure_fn: impl Fn(&mut Dense),
            strategy: LayerInitStrategy,
        ) {
            let seq_len = 4;
            let embed_dim = 8;
            let output_dim = 12;

            assert_input_gradients(
                &move |rng| {
                    let mut dense = Dense::new(embed_dim, output_dim, &strategy, &rng);
                    configure_fn(&mut dense);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, output_dim, &rng);
                    (dense, inputs, target)
                },
                &move |dense, inputs| dense.forward(&inputs).unwrap(),
                &move |dense, inputs, dloss| dense.backward(&inputs, &dloss).unwrap(),
            );
        }
    }
}

pub mod linear {
    use std::{cell::OnceCell, iter};

    use anyhow::{anyhow, Context, Result};
    use itertools::Itertools;
    use serde::{Deserialize, Serialize};

    use crate::ml::{layer::LayerInitStrategy, LayerValues, NodeValue, RngStrategy};

    use self::iter_ext::KnownSizeIterator;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Linear {
        inner: LayerValues,
        stride: usize,
        count: usize,
    }

    impl Linear {
        pub fn new(count: usize, stride: usize) -> Self {
            Self::with_value(count, stride, 0.0)
        }

        pub fn with_dimensions(other: &Self) -> Self {
            Self::with_value(other.count, other.stride, 0.0)
        }

        pub fn with_value(count: usize, stride: usize, value: NodeValue) -> Self {
            let size = count * stride;
            Self {
                inner: LayerValues::new(vec![value; size]),
                stride,
                count,
            }
        }

        pub fn diagonal_iter<'a>(
            size: usize,
            diagonal_value: NodeValue,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue> + 'a> {
            LinearIter {
                inner: (0..(size * size))
                    .map(move |x| (x % size, x / size))
                    .map(move |(i, j)| if i == j { diagonal_value } else { 0.0 }),
                stride: size,
                count: size,
                parent: None,
            }
        }

        pub fn identity(stride: usize) -> Self {
            Self::diagonal_iter(stride, 1.0).collect()
        }

        pub fn from_iter<I: Iterator<Item = NodeValue>>(stride: usize, values: I) -> Result<Self> {
            let inner: LayerValues = values.collect();
            let count = inner.len() / stride;

            if inner.len() != stride * count {
                Err(anyhow!("mismatched values length/stride"))?;
            }

            Ok(Self {
                inner,
                stride,
                count,
            })
        }

        pub fn from_values(values: &[LayerValues]) -> Result<Self> {
            let stride = values.first().context("no values provided")?.len();
            let flattened_inputs = values.into_iter().flat_map(|x| x.iter());
            Linear::from_iter(stride, flattened_inputs.copied())
        }

        pub fn zero(&mut self) {
            self.inner.iter_mut().for_each(|x| *x = 0.0);
        }

        pub fn initialize_as_layer(&mut self, strategy: &LayerInitStrategy, rng: &RngStrategy) {
            strategy.apply(self.inner.iter_mut(), iter::empty(), self.count, &rng);
        }

        pub fn initialize_as_layer_bias(
            &mut self,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) {
            strategy.apply(iter::empty(), self.inner.iter_mut(), self.count, &rng);
        }

        pub fn iter<'a>(&'a self) -> LinearIter<'a, impl Iterator<Item = NodeValue> + 'a> {
            LinearIter {
                inner: self.inner.iter().copied(),
                stride: self.stride,
                count: self.count,
                parent: Some(&self),
            }
        }

        pub fn iter_transpose<'a>(
            &'a self,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue> + 'a> {
            let stride = self.stride;
            let count = self.count;

            let x = (0..self.inner.len()).map(move |idx| {
                let (x, y) = (idx % count, idx / count);
                let (i, j) = (y, x); // transpose dims
                let inner_idx = i + j * stride;
                self.inner[inner_idx]
            });
            LinearIter {
                inner: x,
                stride: count,
                count: stride,
                parent: None,
                // parent: Some(&self), // TODO: could use this?
            }
        }

        pub fn concat<'a>(&'a self, rhs: &'a Linear) -> BoxedLinearIter<'a> {
            assert_eq!(self.count, rhs.count, "mismatched count dimension");
            let self_items = self.inner.chunks_exact(self.stride);
            let rhs_items = rhs.inner.chunks_exact(rhs.stride);
            // TODO: does this need to be boxed?
            BoxedLinearIter {
                inner: Box::new(
                    self_items
                        .zip(rhs_items)
                        .flat_map(|(lhs, rhs)| lhs.iter().chain(rhs).copied())
                        .collect_vec()
                        .into_iter(),
                ),
                stride: self.stride + rhs.stride,
                count: self.count,
                parent: None,
            }
        }

        pub fn split(&self, n: usize) -> Vec<Self> {
            assert_eq!(self.stride % n, 0, "mismatched dimensions");
            let stride = self.stride / n;
            (0..n)
                .map(|i| {
                    let self_items = self.inner.chunks_exact(self.stride);
                    Self {
                        inner: self_items
                            .flat_map(|row| row.into_iter().skip(i * stride).take(stride))
                            .copied()
                            .collect(),
                        stride,
                        count: self.count,
                    }
                })
                .collect()
        }

        pub fn softmax_d(&self, softmax_outputs: &Linear) -> Linear {
            self.iter()
                .sub(
                    softmax_outputs
                        .iter()
                        .dot_product(self.iter())
                        .flatten_sum()
                        .iter()
                        .grow(softmax_outputs.stride()),
                )
                .dot_product(softmax_outputs.iter())
                .collect()
        }

        pub fn matrix_product(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.stride, rhs.count, "mismatched dimensions");

            let is_small = self.count < 64 && rhs.stride < 64;
            if !is_small && self.stride % 16 == 0 && rhs.stride % 16 == 0 {
                self.iter().matrix_product_fast(rhs.iter())
            } else {
                self.iter().matrix_transpose_product(rhs.iter_transpose())
            }
        }

        pub fn matrix_product_rhs_transposed(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.stride, rhs.stride, "mismatched stride dimension");

            let is_small = self.count < 64 && rhs.count < 64;
            if !is_small && self.stride % 16 == 0 && rhs.count % 16 == 0 {
                self.iter().matrix_product_fast(rhs.iter_transpose())
            } else {
                self.iter().matrix_transpose_product(rhs.iter())
            }
        }

        pub fn matrix_product_lhs_transposed(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.count, rhs.count, "mismatched count dimension");

            if self.count % 16 == 0 && rhs.stride % 16 == 0 {
                self.iter_transpose().matrix_product_fast(rhs.iter())
            } else if self.count % 4 == 0 && rhs.stride % 4 == 0 {
                self.iter_transpose().matrix_product_fast_4x4(rhs.iter())
            } else {
                self.iter_transpose()
                    .matrix_transpose_product(rhs.iter_transpose())
            }
        }

        pub fn stride(&self) -> usize {
            self.stride
        }

        pub fn count(&self) -> usize {
            self.count
        }

        pub fn is_finite(&self) -> bool {
            self.inner.iter().all(|x| x.is_finite())
        }

        pub fn to_values(self) -> Vec<LayerValues> {
            self.rows_iter()
                .map(|x| LayerValues::new(x.to_vec()))
                .collect()
        }

        pub fn rows_iter(&self) -> impl Iterator<Item = &[NodeValue]> {
            self.inner.chunks_exact(self.stride)
        }

        pub fn rows_iter_mut(&mut self) -> impl Iterator<Item = &mut [NodeValue]> {
            self.inner.chunks_exact_mut(self.stride)
        }

        pub fn values_iter(&self) -> impl Iterator<Item = (&NodeValue, usize, usize)> {
            self.inner
                .iter()
                .enumerate()
                .map(|(idx, x)| (x, idx % self.stride, idx / self.stride))
        }

        pub fn as_scalar(&self) -> Result<NodeValue> {
            if self.inner.len() == 1 {
                Ok(self.inner[0])
            } else {
                Err(anyhow!(
                    "can not read value of shape = [{}, {}] as scalar",
                    self.count,
                    self.stride
                ))
            }
        }

        pub fn copy_stride_into(&mut self, src: &Self, src_row_idx: usize, dest_row_idx: usize) {
            assert_eq!(self.stride, src.stride, "Mismatched stride dimension");
            assert!(src.count > src_row_idx, "Invalid source row index");
            assert!(self.count > dest_row_idx, "Invalid destination row index");
            let start = src_row_idx * self.stride;
            let end = start + self.stride;
            let src = &src.inner[start..end];

            let start = dest_row_idx * self.stride;
            let end = start + self.stride;
            self.inner[start..end].copy_from_slice(src);
        }

        pub fn as_single_stride(&self) -> Result<LayerValues> {
            if self.count == 1 {
                Ok(self.inner[..self.stride].into())
            } else {
                Err(anyhow!(
                    "can not read shape = [{}, {}] as single stride",
                    self.count,
                    self.stride
                ))
            }
        }

        pub fn to_sum(&self) -> NodeValue {
            self.inner.iter().sum()
        }

        pub fn sum<'a, I: Iterator<Item = &'a Self> + 'a>(
            mut iter: I,
        ) -> Option<BoxedLinearIter<'a>> {
            let first = iter.next()?;
            let mut sum_iter = first.iter().boxed();
            for x in iter {
                sum_iter = sum_iter.add(x.iter()).boxed();
            }
            Some(sum_iter)
        }
    }

    impl std::fmt::Display for Linear {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.rows_iter()
                .fold(&mut f.debug_list(), |list, row| list.entry(&row))
                .finish()
        }
    }

    impl std::str::FromStr for Linear {
        type Err = anyhow::Error;

        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            let lines = s
                .lines()
                .filter_map(|x| Some(x.trim()).filter(|x| !x.is_empty()))
                .collect_vec();

            let count = lines.len();
            let elements: Result<Vec<NodeValue>, _> = lines
                .into_iter()
                .flat_map(|line| line.split_whitespace().map(|x| x.parse::<NodeValue>()))
                .collect();

            let elements = elements.context("invalid numerical found")?;
            let stride = elements.len() / count;

            if stride * count == elements.len() {
                Linear::from_iter(stride, elements.into_iter())
            } else {
                Err(anyhow!("mismatch row length found"))
            }
        }
    }

    impl Default for Linear {
        fn default() -> Self {
            Self {
                inner: LayerValues::new(vec![0.0]),
                stride: 1,
                count: 1,
            }
        }
    }

    pub type BoxedLinearIter<'a> = LinearIter<'a, Box<dyn Iterator<Item = NodeValue> + 'a>>;

    #[must_use = "linear iterators are lazy and do nothing unless consumed"]
    pub struct LinearIter<'a, I> {
        inner: I,
        stride: usize,
        count: usize,
        parent: Option<&'a Linear>,
    }

    impl<'a, I> LinearIter<'a, I>
    where
        I: Iterator<Item = NodeValue> + 'a,
    {
        /// returns underlying data stride dimension size (or 'width')
        pub fn stride(&self) -> usize {
            self.stride
        }
        /// returns underlying data count dimension size (or 'height')
        pub fn count(&self) -> usize {
            self.count
        }
        pub fn boxed(self) -> BoxedLinearIter<'a> {
            LinearIter {
                inner: Box::new(self.inner),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// returns owned result of performing matrix multiplication of (Self * Rhs)
        pub fn matrix_product_fast(
            self,
            rhs: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> Linear {
            if self.stride % 16 == 0 && rhs.stride % 16 == 0 {
                self.matrix_product_fast_n::<16>(rhs)
            } else if self.stride % 4 == 0 && rhs.stride % 4 == 0 {
                self.matrix_product_fast_n::<4>(rhs)
            } else {
                unimplemented!("no remained implementation")
            }
        }
        /// returns owned result of performing matrix multiplication of (Self * Rhs)
        pub fn matrix_product_fast_4x4(
            self,
            rhs: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> Linear {
            self.matrix_product_fast_n::<4>(rhs)
        }
        /// returns owned result of performing matrix multiplication of (Self * Rhs)
        pub fn matrix_product_fast_n<const N: usize>(
            self,
            rhs: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> Linear {
            assert_eq!(self.stride % N, 0, "mismatched lhs stride alignment");
            assert_eq!(rhs.stride % N, 0, "mismatched rhs stride alignment");
            let m = self.count;
            let n = rhs.stride;
            let k = {
                assert_eq!(self.stride, rhs.count, "mismatched stride dimension");
                self.stride // equal to rhs.count
            };
            let self_oc = OnceCell::new();
            let rhs_oc = OnceCell::new();
            let a = self.get_or_init_inner(&self_oc);
            let b = rhs.get_or_init_inner(&rhs_oc);

            // a[k * (row) + (col)] where 0 <= row < m, and 0 <= col < k
            // b[n * (row) + (col)] where 0 <= row < k, and 0 <= col < n
            // c[n * (row) + (col)] where 0 <= row < m, and 0 <= col < n

            let stride = n;
            let count = m;
            let mut c = vec![0.0; stride * count];

            for (c_row, a_row) in c.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
                for (a_chunk, b_chunk_rows) in a_row.chunks_exact(N).zip(b.chunks_exact(n * N)) {
                    for (&a, b_chunk_row) in a_chunk.iter().zip(b_chunk_rows.chunks_exact(n)) {
                        for (c_block, b_chunk) in
                            c_row.chunks_exact_mut(N).zip(b_chunk_row.chunks(N))
                        {
                            for (c, &b) in c_block.iter_mut().zip(b_chunk.iter()) {
                                *c = unsafe { std::intrinsics::fmaf64(a, b, *c) };
                                // *c += a * b;
                            }
                        }
                    }
                }
            }

            Linear {
                inner: LayerValues::new(c),
                stride,
                count,
            }
        }

        /// returns owned result of performing matrix multiplication of (Self * Rhs.T)
        pub fn matrix_transpose_product(
            self,
            rhs_transpose: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> Linear {
            assert_eq!(
                self.stride, rhs_transpose.stride,
                "mismatched stride dimension"
            );
            let self_oc = OnceCell::new();
            let rhs_oc = OnceCell::new();
            let self_vec = self
                .parent
                .map(|x| x.inner.as_ref())
                .unwrap_or_else(|| self_oc.get_or_init(|| self.inner.collect_vec()));
            let rhs_vec = rhs_transpose
                .parent
                .map(|x| x.inner.as_ref())
                .unwrap_or_else(|| rhs_oc.get_or_init(|| rhs_transpose.inner.collect_vec()));

            let self_strides = self_vec.chunks_exact(self.stride);
            let rhs_strides = rhs_vec.chunks_exact(rhs_transpose.stride);

            let len = rhs_transpose.count * self.count;
            let mut inner = LayerValues::new(vec![0.0; len]);

            self_strides
                .into_iter()
                .flat_map(|a| {
                    rhs_strides.clone().map(move |b| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(a, b)| a * b)
                            .sum::<NodeValue>()
                    })
                })
                .zip(inner.iter_mut())
                .for_each(|(x, i)| *i = x);

            Linear {
                inner,
                stride: rhs_transpose.count,
                count: self.count,
            }
        }

        /// point-wise multiplication
        pub fn dot_product(
            self,
            other: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, other.stride, "mismatched stride dimension");
            assert_eq!(self.count, other.count, "mismatched count dimension");
            LinearIter {
                inner: self.inner.zip(other.inner).map(|(x, y)| x * y),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise division
        pub fn div(
            self,
            rhs: LinearIter<'a, impl Iterator<Item = NodeValue>>,
            epsilon: Option<NodeValue>,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, rhs.stride, "mismatched stride dimension");
            assert_eq!(self.count, rhs.count, "mismatched count dimension");
            let e = epsilon.unwrap_or(0.0);
            LinearIter {
                inner: self.inner.zip(rhs.inner).map(move |(x, y)| x / (y + e)),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise addition
        pub fn add(
            self,
            other: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, other.stride, "mismatched stride dimension");
            assert_eq!(self.count, other.count, "mismatched count dimension");
            LinearIter {
                inner: self.inner.zip(other.inner).map(|(x, y)| x + y),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise addition by scalar constant
        pub fn add_scalar(self, rhs: NodeValue) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x + rhs),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise subtraction
        pub fn sub(
            self,
            rhs: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, rhs.stride, "mismatched stride dimension");
            assert_eq!(self.count, rhs.count, "mismatched count dimension");
            LinearIter {
                inner: self.inner.zip(rhs.inner).map(|(x, y)| x - y),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise subtraction by scalar constant
        pub fn sub_scalar(self, rhs: NodeValue) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x - rhs),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise multiplication by scalar constant
        pub fn multiply_scalar(
            self,
            rhs: NodeValue,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x * rhs),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise raise to the power of scalar integer constant
        pub fn powi_scalar(self, n: i32) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x.powi(n)),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise raise to the power of scalar floting point constant
        pub fn powf_scalar(self, n: NodeValue) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x.powf(n)),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise round to fixed point decimal
        pub fn round(self, decimals: u32) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            let mul = 10u32.pow(decimals) as NodeValue;
            LinearIter {
                inner: self.inner.map(move |x| (x * mul).round() / mul),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise absolute value computation
        pub fn abs(self) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x.abs()),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// point-wise square root value computation
        pub fn sqrt(self) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            LinearIter {
                inner: self.inner.map(move |x| x.sqrt()),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        pub fn dropout_mask(&self, dropout_rate: NodeValue, rng: RngStrategy) -> Linear {
            assert!(
                dropout_rate <= 1.0 && dropout_rate >= 0.0,
                "invalid dropout rate"
            );
            let mut linear = Linear::with_value(self.count, self.stride, 1.0);

            if dropout_rate != 0.0 {
                let normalized_value = 1.0 / (1.0 - dropout_rate);
                linear.rows_iter_mut().flat_map(|x| x).for_each(move |x| {
                    *x = if rng.rand() < dropout_rate {
                        0.0
                    } else {
                        normalized_value
                    }
                });
            }

            linear
        }
        /// point-wise mask-based value overwrite computation.
        /// Note: a value at given position is set to masked_value only where the value of the mask is 0.0
        pub fn set_mask(
            self,
            mask: LinearIter<'a, impl Iterator<Item = NodeValue>>,
            masked_value: NodeValue,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, mask.stride, "mismatched stride dimension");
            assert_eq!(self.count, mask.count, "mismatched count dimensions");
            LinearIter {
                inner: Box::new(self.inner.zip(mask.inner).map(move |(x, mask)| {
                    if mask == 0.0 {
                        masked_value
                    } else {
                        x
                    }
                })),
                stride: self.stride,
                count: self.count,
                parent: None,
            }
        }
        /// extends stride dimension by copying duplicating column values
        /// Note: stride dimension must be equal to 1
        pub fn grow(self, stride: usize) -> LinearIter<'a, impl Iterator<Item = NodeValue>> {
            assert_eq!(self.stride, 1, "can only grow when stride dimension = 1");
            assert_ne!(stride, 0, "invalid stride dimension");
            LinearIter {
                inner: self.inner.flat_map(move |x| iter::repeat(x).take(stride)),
                stride,
                count: self.count,
                parent: None,
            }
        }
        /// extends count dimension by copying duplicating row values
        /// Note: count dimension must be equal to 1
        pub fn stack(self, count: usize) -> BoxedLinearIter<'a> {
            assert_eq!(self.count, 1, "can only stack when count dimension = 1");
            assert_ne!(count, 0, "invalid stride dimension");
            LinearIter {
                inner: match self.parent {
                    Some(parent) => Box::new(
                        parent
                            .inner
                            .iter()
                            .copied()
                            .cycle()
                            .take(self.stride * count),
                    ),
                    None => Box::new(
                        self.inner
                            .collect_vec()
                            .into_iter()
                            .cycle()
                            .take(self.stride * count),
                    ),
                },

                stride: self.stride,
                count,
                parent: None,
            }
        }

        pub fn apply_one(self, lhs_inputs: &[NodeValue]) -> LayerValues {
            assert_eq!(self.count, lhs_inputs.len(), "mismatched dimensions");

            self.inner
                .chunks(self.stride)
                .into_iter()
                .zip(lhs_inputs)
                .flat_map(|(row, input)| row.map(move |x| x * input))
                .collect()
        }
        pub fn apply_gradients(
            self,
            grads: LinearIter<'a, impl Iterator<Item = NodeValue>>,
            learn_rate: NodeValue,
        ) -> Linear {
            self.sub(grads.multiply_scalar(learn_rate)).collect()
        }
        pub fn softmax(self) -> Linear {
            let inner_vec = self.inner.collect_vec();
            let mut exp_counts = inner_vec
                .chunks_exact(self.stride)
                .flat_map(|chunk| {
                    let max = chunk
                        .iter()
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .cloned()
                        .unwrap_or_default();
                    chunk.iter().map(move |&x| {
                        if x != NodeValue::NEG_INFINITY {
                            (x - max).exp()
                        } else {
                            0.0
                        }
                    })
                })
                .collect_vec();
            let inner = {
                exp_counts.chunks_exact_mut(self.stride).for_each(|chunk| {
                    let sum = chunk.iter().sum::<NodeValue>();
                    let sum = if sum != 0.0 { sum } else { 1e-8 };
                    chunk.iter_mut().for_each(|x| *x = *x / sum);
                });
                exp_counts.into()
            };
            Linear {
                inner,
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn flatten_sum(self) -> Linear {
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .map(|x| x.sum::<NodeValue>())
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        pub fn flatten_mean(self) -> Linear {
            let stride = self.stride as NodeValue;
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .map(|x| x.sum::<NodeValue>() / stride)
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        pub fn flatten_stddev_corrected(
            self,
            mean: LinearIter<'a, impl Iterator<Item = NodeValue>>,
        ) -> Linear {
            self.flatten_stddev(mean, true)
        }
        pub fn flatten_stddev(
            self,
            mean: LinearIter<'a, impl Iterator<Item = NodeValue>>,
            corrected: bool,
        ) -> Linear {
            assert_eq!(self.count, mean.count, "mismatched count dimension");
            assert_eq!(mean.stride, 1, "invalid mean stride dimension");
            if self.stride == 1 {
                return Linear::with_value(self.count, 1, 0.0);
            }
            let factor = if corrected {
                ((self.stride - 1) as NodeValue).powf(-0.5)
            } else {
                (self.stride as NodeValue).powf(-0.5)
            };
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .zip(mean.inner)
                    .map(|(x, mean)| {
                        x.map(|x| (x - mean).powi(2)).sum::<NodeValue>().sqrt() * factor
                    })
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        fn get_or_init_inner<'b>(self, cell: &'b OnceCell<Vec<NodeValue>>) -> &'b Vec<NodeValue>
        where
            'a: 'b,
        {
            self.parent
                .map(|x| x.inner.as_ref())
                .unwrap_or_else(move || cell.get_or_init(|| self.inner.collect_vec()))
        }
        pub fn collect(self) -> Linear {
            Linear {
                inner: self.inner.with_size(self.stride * self.count).collect(),
                stride: self.stride,
                count: self.count,
            }
        }
    }

    mod iter_ext {
        use crate::ml::NodeValue;

        pub trait KnownSizeIterator: Iterator<Item = NodeValue> {
            fn with_size(self, size: usize) -> KnownSizedIter<Self>
            where
                Self: Sized,
            {
                KnownSizedIter { inner: self, size }
            }
        }

        pub struct KnownSizedIter<I> {
            size: usize,
            inner: I,
        }

        impl<I: Iterator<Item = NodeValue>> Iterator for KnownSizedIter<I> {
            type Item = NodeValue;

            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.size, Some(self.size))
            }
        }

        impl<I: Iterator<Item = NodeValue>> ExactSizeIterator for KnownSizedIter<I> {
            fn len(&self) -> usize {
                self.size
            }
        }

        impl<I: Iterator<Item = NodeValue>> KnownSizeIterator for I {}
    }

    pub fn mat_mul(
        stride: usize,
        count: usize,
        inner_dim: usize,
        a: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        const N: usize = 32;
        let (n, k) = (stride, inner_dim);
        let mut c = vec![0.0; stride * count];

        for (c_row, a_row) in c.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
            for (a_chunk, b_chunk_rows) in a_row.chunks_exact(N).zip(b.chunks_exact(n * N)) {
                for (&a, b_chunk_row) in a_chunk.iter().zip(b_chunk_rows.chunks_exact(n)) {
                    for (c_block, b_chunk) in c_row.chunks_exact_mut(N).zip(b_chunk_row.chunks(N)) {
                        // assert!(c_block)
                        for (c, &b) in c_block.iter_mut().zip(b_chunk.iter()) {
                            *c = a.mul_add(b, *c);
                            // *c += a * b;
                        }
                    }
                }
            }
        }
        c
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn can_linear_perform_mat_mul() {
            let a = Linear::from_iter(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [10.0, 11.0, 20.0, 21.0, 30.0, 31.0].into_iter()).unwrap();
            let y = a.matrix_product(&b);

            let expected = Linear::from_iter(2, [140.0, 146.0, 320.0, 335.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_fast_mat_mul() {
            let (n, k) = (64, 32);
            let mut a = Linear::with_value(n, k, 0.0);
            a.rows_iter_mut().enumerate().for_each(|(i, row)| {
                row.split_at_mut(i.min(row.len() - 1))
                    .0
                    .iter_mut()
                    .for_each(|x| *x = 1.0)
            });

            let mut b = Linear::with_value(k, n, 0.0);
            b.rows_iter_mut().enumerate().for_each(|(i, row)| {
                row.split_at_mut(i.min(row.len() - 1))
                    .1
                    .iter_mut()
                    .for_each(|x| *x = 1.0)
            });

            let expected = a.iter().matrix_transpose_product(b.iter_transpose());
            let y = a.iter().matrix_product_fast(b.iter());
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_fast_mat_mul_explicit() {
            let a = r#"
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            "#;

            let b = r#"
                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
                0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
                0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
                0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
                0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
                0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
                0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
                0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
                0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1
                0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
            "#;

            let matrix_a: Linear = a.parse().unwrap();
            let matrix_b: Linear = b.parse().unwrap();

            let expected = r#"
                0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
                1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
                1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
                1  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3
                1  2  3  4  4  4  4  4  4  4  4  4  4  4  4  4
                1  2  3  4  5  5  5  5  5  5  5  5  5  5  5  5
                1  2  3  4  5  6  6  6  6  6  6  6  6  6  6  6
                1  2  3  4  5  6  7  7  7  7  7  7  7  7  7  7
                1  2  3  4  5  6  7  8  8  8  8  8  8  8  8  8
                1  2  3  4  5  6  7  8  9  9  9  9  9  9  9  9
                1  2  3  4  5  6  7  8  9  10 10 10 10 10 10 10
                1  2  3  4  5  6  7  8  9  10 11 11 11 11 11 11
                1  2  3  4  5  6  7  8  9  10 11 12 12 12 12 12
                1  2  3  4  5  6  7  8  9  10 11 12 13 13 13 13
                1  2  3  4  5  6  7  8  9  10 11 12 13 14 14 14
                1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 15
            "#;
            let expected_matrix: Linear = expected.parse().unwrap();

            let c = matrix_a
                .iter()
                .matrix_transpose_product(matrix_b.iter_transpose());
            assert_eq!(expected_matrix, c);

            let c = matrix_a.iter().matrix_product_fast(matrix_b.iter());
            assert_eq!(expected_matrix, c);
        }

        #[test]
        fn can_linear_perform_transpose() {
            let a = Linear::from_iter(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter()).unwrap();
            let y = a.iter_transpose().collect();

            let expected =
                Linear::from_iter(2, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_add_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 6.0, 7.0, 8.0].into_iter()).unwrap();
            let y = a.iter().add(b.iter()).collect();

            let expected = Linear::from_iter(2, [6.0, 8.0, 10.0, 12.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_sub_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 4.0, 3.0, 2.0].into_iter()).unwrap();
            let y = a.iter().sub(b.iter()).collect();

            let expected = Linear::from_iter(2, [-4.0, -2.0, 0.0, 2.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_div_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 4.0, 3.0, 2.0].into_iter()).unwrap();
            let y = a.iter().div(b.iter(), None).collect();

            let expected = Linear::from_iter(2, [0.2, 0.5, 1.0, 2.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_dot_product() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 6.0, 7.0, 8.0].into_iter()).unwrap();
            let y = a.iter().dot_product(b.iter()).collect();

            let expected = Linear::from_iter(2, [5.0, 12.0, 21.0, 32.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_scalar_ops() {
            let x = Linear::from_iter(2, [1.0, 2.0, 1.0, 0.0].into_iter()).unwrap();

            let y = x.iter().multiply_scalar(4.0).collect();
            let expected = Linear::from_iter(2, [4.0, 8.0, 4.0, 0.0].into_iter()).unwrap();
            assert_eq!(expected, y);

            let y = x.iter().add_scalar(4.0).collect();
            let expected = Linear::from_iter(2, [5.0, 6.0, 5.0, 4.0].into_iter()).unwrap();
            assert_eq!(expected, y);

            let y = x.iter().sub_scalar(4.0).collect();
            let expected = Linear::from_iter(2, [-3.0, -2.0, -3.0, -4.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_flatten_sum() {
            let x = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = x.iter().flatten_sum();

            let expected = Linear::from_iter(1, [3.0, 7.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_sum_rows_and_keep_shape() {
            let x = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = x.iter().flatten_sum().iter().grow(2).collect();

            let expected = Linear::from_iter(2, [3.0, 3.0, 7.0, 7.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_mean_stride_values() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = a.iter().flatten_mean();

            let expected = Linear::from_iter(1, [1.5, 3.5].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_stddev_stride_values() {
            let a = Linear::from_iter(5, [1.0, 2.0, 3.0, 4.0, 5.0].into_iter()).unwrap();
            let a_mean = a.iter().flatten_mean();

            let corrected_y = a.iter().flatten_stddev_corrected(a_mean.iter());
            let rounded_y = corrected_y.iter().round(4).collect();
            let expected = Linear::from_iter(1, [1.5811].into_iter()).unwrap();
            assert_eq!(expected, rounded_y);

            let uncorrected_y = a.iter().flatten_stddev(a_mean.iter(), false);
            let rounded_y = uncorrected_y.iter().round(4).collect();
            let expected = Linear::from_iter(1, [1.4142].into_iter()).unwrap();
            assert_eq!(expected, rounded_y);
        }

        #[test]
        fn can_linear_perform_mean_all_values_and_keep_shape() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let mean_a = a.iter().flatten_mean().iter_transpose().flatten_mean();
            let y = mean_a.iter().grow(a.stride()).stack(a.count()).collect();

            let expected = Linear::from_iter(2, [2.5, 2.5, 2.5, 2.5].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_softmax() {
            let x = Linear::from_iter(
                2,
                [1.0, 1.0, 0.0, 0.0, -100.0, -100.0, 100.0, 100.0].into_iter(),
            )
            .unwrap();
            let y = x.iter().softmax();

            let expected =
                Linear::from_iter(2, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_masked_softmax() {
            let x = Linear::from_iter(
                2,
                [-1.0, 1.0, 0.0, 0.0, 25.0, 175.0, -25.0, -75.0].into_iter(),
            )
            .unwrap();
            let mask =
                Linear::from_iter(2, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0].into_iter()).unwrap();

            let y = x
                .iter()
                .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                .softmax();

            let expected =
                Linear::from_iter(2, [1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_diagonal_masked_softmax() {
            let x = Linear::from_iter(
                3,
                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0].into_iter(),
            )
            .unwrap();
            let mask = Linear::identity(3);

            let y = x
                .iter()
                .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                .softmax();

            let const_1_3 = 1.0 / 3.0;
            let expected = Linear::from_iter(
                3,
                [
                    1.0, 0.0, 0.0, 0.5, 0.5, 0.0, const_1_3, const_1_3, const_1_3,
                ]
                .into_iter(),
            )
            .unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_create_identity_matix() {
            let x = Linear::identity(3);
            let expected =
                Linear::from_iter(3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].into_iter())
                    .unwrap();
            assert_eq!(expected, x);
        }

        #[test]
        fn can_linear_split_and_concat_again() {
            let x = Linear::from_values(&[
                [11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0].into(),
                [11.1, 12.1, 13.1, 21.1, 22.1, 23.1, 31.1, 32.1, 33.1].into(),
                [11.2, 12.2, 13.2, 21.2, 22.2, 23.2, 31.2, 32.2, 33.2].into(),
            ])
            .unwrap();

            let split = x.split(3);
            let expected_split = Linear::from_iter(
                3,
                [11.0, 12.0, 13.0, 11.1, 12.1, 13.1, 11.2, 12.2, 13.2].into_iter(),
            )
            .unwrap();
            assert_eq!(3, split.len());
            assert_eq!(expected_split, split[0]);

            let joined = &split[0].concat(&split[1]).collect();
            let joined = joined.concat(&split[2]).collect();
            let expected_joined = x;

            assert_eq!(expected_joined, joined);
        }
    }
}

pub mod params {
    use std::{
        fmt::Debug,
        sync::{Arc, Mutex, RwLock},
    };

    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    use crate::{lazy_opt, ml::NodeValue};

    use super::{
        linear::{BoxedLinearIter, Linear, LinearIter},
        solver::{source::OptimizerSource, Optimizer},
    };

    pub trait TrainableParameter<P: keys::TrainableParameterKey> {
        fn store(&self) -> Option<ParameterStore>;
        fn param_mut(&mut self) -> Option<&mut TrainableLinear>;
        fn queue_gradients(&self, _: P, gradients: BoxedLinearIter) {
            if let Some(store) = self.store() {
                store.add_gradients(gradients);
            }
        }
        fn apply_param_gradients<T: OptimizerSource>(&mut self, _: P, optimizer: &T) -> Result<()> {
            if let Some(store) = self.store() {
                store.apply_param_gradients(optimizer, self.param_mut())?;
            }
            Ok(())
        }
    }

    pub trait TrainableCollection<P: keys::TrainableParameterKey> {
        fn store(&self, index: usize) -> Option<ParameterStore>;
        fn param_mut(&mut self) -> Option<&mut [TrainableLinear]>;
        fn queue_gradients(&self, _: P, index: usize, gradients: BoxedLinearIter) {
            if let Some(store) = self.store(index) {
                store.add_gradients(gradients);
            }
        }
        fn apply_param_gradients<T: OptimizerSource>(&mut self, _: P, optimizer: &T) -> Result<()> {
            let count = self.param_mut().map(|x| x.len()).unwrap_or_default();
            for index in 0..count {
                if let Some(store) = self.store(index) {
                    store.apply_param_gradients(
                        optimizer,
                        self.param_mut().and_then(|x| x.get_mut(index)),
                    )?;
                }
            }
            Ok(())
        }
    }

    #[derive(Debug, Default, PartialEq, Clone, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct TrainableLinear {
        value: Linear,
        #[serde(skip)]
        parameters: ParameterStore,
    }

    impl TrainableLinear {
        pub fn parameters(&self) -> ParameterStore {
            self.parameters.clone()
        }

        pub fn value(&self) -> &Linear {
            &self.value
        }

        pub fn value_mut(&mut self) -> &mut Linear {
            &mut self.value
        }

        pub fn iter<'a>(&'a self) -> LinearIter<'a, impl Iterator<Item = NodeValue> + 'a> {
            self.value.iter()
        }

        pub fn iter_transpose<'a>(
            &'a self,
        ) -> LinearIter<'a, impl Iterator<Item = NodeValue> + 'a> {
            self.value.iter_transpose()
        }

        pub fn stride(&self) -> usize {
            self.value.stride()
        }

        pub fn count(&self) -> usize {
            self.value.count()
        }
    }

    impl From<Linear> for TrainableLinear {
        fn from(value: Linear) -> Self {
            Self {
                value,
                parameters: Default::default(),
            }
        }
    }

    // TODO: evaluate and possibly remove queue.. left over from RefCell store
    #[derive(Debug, Default, Clone)]
    pub struct ParameterStore(Arc<RwLock<Linear>>, Arc<Mutex<Vec<Linear>>>);

    impl PartialEq for ParameterStore {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl ParameterStore {
        fn add_gradients(&self, gradients: BoxedLinearIter) {
            let mut store = match self.0.try_write() {
                Ok(store) => store,
                Err(_) => {
                    self.enqueue_gradient(gradients);
                    return;
                }
            };
            let (stride, count) = (store.stride(), store.count());
            if stride == 1 && count == 1 {
                if stride != gradients.stride() || count != gradients.count() {
                    *store = Linear::new(gradients.count(), gradients.stride());
                }
            }

            *store = store.iter().add(gradients).collect();
        }
        fn apply_param_gradients<T: OptimizerSource>(
            &self,
            optimizer: &T,
            mut param: Option<&mut TrainableLinear>,
        ) -> Result<()> {
            for pending in self.dequeue_all() {
                self.add_gradients(pending.iter().boxed());
            }
            self.with_gradients(move |gradients| {
                if let Some(param) = param.as_mut() {
                    let param = param.value_mut();
                    let mut opt_param = lazy_opt!(optimizer, param);
                    opt_param.update(param, gradients)?;
                }
                Ok(())
            })?;
            Ok(())
        }
        fn with_gradients(&self, mut action: impl FnMut(&Linear) -> Result<()>) -> Result<()> {
            let mut gradients = self.0.write().unwrap();
            if let Ok(true) = gradients.as_scalar().map(|x| x == 0.0) {
                return Ok(());
            }
            action(&*gradients)?;
            gradients.zero();
            Ok(())
        }
        fn enqueue_gradient(&self, gradients: BoxedLinearIter) {
            self.1.lock().unwrap().push(gradients.collect());
        }
        fn dequeue_all(&self) -> Vec<Linear> {
            self.1.lock().unwrap().drain(..).collect()
        }
        pub fn peek_gradients<T>(&self, action: impl FnOnce(&Linear) -> T) -> T {
            let gradients = self.0.read().unwrap();
            action(&*gradients)
        }
    }

    pub mod keys {
        use std::{fmt::Debug, hash::Hash};

        pub use attention::*;
        pub use dense::*;
        pub use embedding::*;
        pub use norm::*;

        pub trait TrainableParameterKey: Hash + Debug {}

        pub mod attention {
            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct AttentionKeyWeights;
            impl super::TrainableParameterKey for AttentionKeyWeights {}

            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct AttentionQueryWeights;
            impl super::TrainableParameterKey for AttentionQueryWeights {}

            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct AttentionValueWeights;
            impl super::TrainableParameterKey for AttentionValueWeights {}
        }
        pub mod dense {

            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct DenseWeight;
            impl super::TrainableParameterKey for DenseWeight {}

            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct DenseBias;
            impl super::TrainableParameterKey for DenseBias {}
        }
        pub mod norm {
            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct LayerNormalizationBeta;
            impl super::TrainableParameterKey for LayerNormalizationBeta {}

            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct LayerNormalizationGamma;
            impl super::TrainableParameterKey for LayerNormalizationGamma {}
        }
        pub mod embedding {
            #[derive(Debug, Clone, Copy, Hash, PartialEq)]
            pub struct EmbeddingVector;
            impl super::TrainableParameterKey for EmbeddingVector {}
        }
    }
}

pub mod solver {
    use anyhow::{anyhow, Result};

    use crate::ml::NodeValue;

    use self::source::{DefaultOptimizerCache, DynamicOptimizerFactory, OptimizerSource};

    use super::linear::Linear;

    pub trait Optimizer {
        fn update(&mut self, target: &mut Linear, dloss_dtarget: &Linear) -> Result<()>;
    }

    pub struct AdamOptimizer {
        momentum: Linear,
        rms: Linear,
        beta: (NodeValue, NodeValue),
        epsilon: NodeValue,
        eta: NodeValue,
        t: u64,
    }

    impl AdamOptimizer {
        pub fn new(param_count: usize, param_dimension: usize, learn_rate: NodeValue) -> Self {
            Self::new_builder(param_count, param_dimension)
                .with_eta(learn_rate)
                .build()
        }

        pub fn new_builder(
            param_count: usize,
            param_dimension: usize,
        ) -> builder::AdamOptimizerBuilder {
            builder::AdamOptimizerBuilder::new(param_count, param_dimension)
        }

        pub fn set_eta(&mut self, eta: NodeValue) {
            self.eta = eta;
        }

        pub fn new_cache(
            learn_rate: NodeValue,
        ) -> DefaultOptimizerCache<DynamicOptimizerFactory<Self>, Self> {
            DefaultOptimizerCache::new(DynamicOptimizerFactory::new(
                move |param_count, param_dimension| {
                    Self::new(param_count, param_dimension, learn_rate)
                },
            ))
        }
    }

    impl Optimizer for AdamOptimizer {
        fn update(&mut self, target: &mut Linear, dloss_dtarget: &Linear) -> Result<()> {
            let beta1 = self.beta.0;
            let beta2 = self.beta.1;
            let t = self.t as NodeValue;

            let gradient_squared = dloss_dtarget.iter().powf_scalar(2.0);

            self.momentum = self
                .momentum
                .iter()
                .multiply_scalar(beta1)
                .add(dloss_dtarget.iter().multiply_scalar(1.0 - beta1))
                .collect();

            self.rms = self
                .rms
                .iter()
                .multiply_scalar(beta2)
                .add(gradient_squared.multiply_scalar(1.0 - beta2))
                .collect();

            let momentum_corrected = &self
                .momentum
                .iter()
                .multiply_scalar(1.0 / (1.0 - beta1.powf(t)))
                .collect();

            let rms_corrected = &self
                .rms
                .iter()
                .multiply_scalar(1.0 / (1.0 - beta2.powf(t)))
                .collect();

            let epsilon = Some(self.epsilon);
            let eta_correction = momentum_corrected
                .iter()
                .div(rms_corrected.iter().powf_scalar(0.5), epsilon)
                .multiply_scalar(-self.eta);

            let next_value = target.iter().add(eta_correction).collect();
            if !next_value.is_finite() {
                Err(anyhow!("failed to update target: invalid gradients"))?;
            }

            self.t += 1;
            *target = next_value;
            Ok(())
        }
    }

    pub struct RMSpropOptimizer {
        cache: Linear,
        gamma: NodeValue,
        epsilon: NodeValue,
        eta: NodeValue,
    }

    impl RMSpropOptimizer {
        pub fn new(param_count: usize, param_dimension: usize) -> Self {
            Self {
                cache: Linear::new(param_count, param_dimension),
                gamma: 0.9,
                epsilon: 1e-8,
                eta: 0.001,
            }
        }

        pub fn with_gamma(mut self, gamma: NodeValue) -> Self {
            self.gamma = gamma;
            self
        }

        pub fn with_epsilon(mut self, epsilon: NodeValue) -> Self {
            self.epsilon = epsilon;
            self
        }

        pub fn with_eta(mut self, eta: NodeValue) -> Self {
            self.eta = eta;
            self
        }

        pub fn new_cache(learn_rate: NodeValue) -> impl OptimizerSource {
            DefaultOptimizerCache::new(DynamicOptimizerFactory::new(
                move |param_count, param_dimension| {
                    Self::new(param_count, param_dimension).with_eta(learn_rate)
                },
            ))
        }
    }

    impl Optimizer for RMSpropOptimizer {
        fn update(&mut self, target: &mut Linear, dloss_dtarget: &Linear) -> Result<()> {
            let gamma = self.gamma;
            let epsilon = self.epsilon;
            let eta = self.eta;

            let gradient_squared = dloss_dtarget.iter().powf_scalar(2.0);
            self.cache = self
                .cache
                .iter()
                .multiply_scalar(gamma)
                .add(gradient_squared.multiply_scalar(1.0 - gamma))
                .collect();

            let eta_correction = dloss_dtarget
                .iter()
                .div(self.cache.iter().powf_scalar(0.5), Some(epsilon))
                .multiply_scalar(-eta);

            let next_value = target.iter().add(eta_correction).collect();
            if !next_value.is_finite() {
                Err(anyhow!("failed to update target: invalid gradients"))?;
            }

            *target = next_value;
            Ok(())
        }
    }

    pub struct SGDOptimizer {
        learn_rate: NodeValue,
    }

    impl SGDOptimizer {
        pub fn new(learn_rate: NodeValue) -> Self {
            Self { learn_rate }
        }

        pub fn new_cache(learn_rate: NodeValue) -> impl OptimizerSource {
            DefaultOptimizerCache::new(DynamicOptimizerFactory::new(move |_, _| {
                Self::new(learn_rate)
            }))
        }
    }

    impl Optimizer for SGDOptimizer {
        fn update(&mut self, target: &mut Linear, dloss_dtarget: &Linear) -> Result<()> {
            let next_value = target
                .iter()
                .apply_gradients(dloss_dtarget.iter(), self.learn_rate);
            if !next_value.is_finite() {
                Err(anyhow!("failed to update target: invalid gradients"))?;
            }

            *target = next_value;
            Ok(())
        }
    }

    pub mod source {
        use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

        use tracing::debug;

        use super::{lazy::LazyOptimizer, Optimizer};

        pub trait OptimizerSource: Clone
        where
            <Self as OptimizerSource>::Optimizer: Optimizer,
        {
            type Optimizer;

            fn create(
                &self,
                param_count: usize,
                param_dimension: usize,
                instance_name: Option<String>,
            ) -> Self::Optimizer;
            fn create_lazy(&self) -> LazyOptimizer<Self::Optimizer>;
            fn create_lazy_named(&self, instance_name: String) -> LazyOptimizer<Self::Optimizer>;
            fn with_next_index(&self) -> Box<Self> {
                self.with_index(1)
            }
            fn with_index(&self, index: usize) -> Box<Self> {
                _ = index;
                Box::new(self.clone())
            }
        }

        pub trait DefaultOptimizerFactory<O>: Clone {
            fn create(&self, param_count: usize, param_dimension: usize) -> O;
        }

        pub struct DefaultOptimizerCache<F, O>
        where
            F: DefaultOptimizerFactory<O>,
        {
            factory: F,
            instances: Rc<RefCell<HashMap<String, Rc<RefCell<O>>>>>,
            depth: usize,
        }

        impl<F, O> Clone for DefaultOptimizerCache<F, O>
        where
            F: DefaultOptimizerFactory<O>,
        {
            fn clone(&self) -> Self {
                Self {
                    factory: self.factory.clone(),
                    instances: self.instances.clone(),
                    depth: self.depth.clone(),
                }
            }
        }

        impl<'a, F: DefaultOptimizerFactory<O>, O> DefaultOptimizerCache<F, O> {
            pub fn new(factory: F) -> Self {
                Self {
                    factory,
                    instances: Rc::new(RefCell::new(HashMap::new())),
                    depth: 0,
                }
            }

            fn new_instance(&self, param_count: usize, param_dimension: usize) -> O {
                self.factory.create(param_count, param_dimension)
            }

            pub fn for_each_mut<G: FnMut(&mut O) -> ()>(&mut self, mut f: G) {
                self.instances
                    .borrow_mut()
                    .values()
                    .for_each(|x| f(&mut (*x.borrow_mut())))
            }
        }

        impl<F, O> OptimizerSource for DefaultOptimizerCache<F, O>
        where
            F: DefaultOptimizerFactory<O>,
            O: Optimizer + 'static,
        {
            type Optimizer = OptimizerCacheEntry<O>;

            fn create(
                &self,
                param_count: usize,
                param_dimension: usize,
                instance_name: Option<String>,
            ) -> Self::Optimizer {
                let mut name = instance_name.unwrap_or_else(|| {
                    let id = self.instances.borrow().len();
                    format!("unnamed_{id}_")
                });
                name.extend(self.depth.to_string().chars());
                let value = self
                    .instances
                    .borrow_mut()
                    .entry(name)
                    .or_insert_with_key(|key| {
                        debug!("Created new optimiser instance with key='{}'", key);
                        let value = self.new_instance(param_count, param_dimension);
                        let value = RefCell::new(value);
                        Rc::new(value)
                    })
                    .clone();
                OptimizerCacheEntry(value)
            }

            fn create_lazy(&self) -> LazyOptimizer<Self::Optimizer> {
                LazyOptimizer::new(|param_count, param_dimension| {
                    self.create(param_count, param_dimension, None)
                })
            }

            fn create_lazy_named(&self, instance_name: String) -> LazyOptimizer<Self::Optimizer> {
                LazyOptimizer::new(move |param_count, param_dimension| {
                    self.create(param_count, param_dimension, Some(instance_name))
                })
            }

            fn with_index(&self, index: usize) -> Box<Self> {
                assert!(index < 100);
                Box::new(Self {
                    factory: self.factory.clone(),
                    instances: self.instances.clone(),
                    depth: (self.depth * 100) + index,
                })
            }
        }

        pub struct OptimizerCacheEntry<O>(Rc<RefCell<O>>);

        impl<T: Optimizer> Optimizer for OptimizerCacheEntry<T> {
            fn update(
                &mut self,
                target: &mut crate::ml::transformer::linear::Linear,
                dloss_dtarget: &crate::ml::transformer::linear::Linear,
            ) -> anyhow::Result<()> {
                self.0.borrow_mut().update(target, dloss_dtarget)
            }
        }

        pub struct DynamicOptimizerFactory<O> {
            inner: Arc<dyn Fn(usize, usize) -> O>,
        }

        impl<O> Clone for DynamicOptimizerFactory<O> {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<O> DynamicOptimizerFactory<O> {
            pub fn new<F: Fn(usize, usize) -> O + 'static>(inner: F) -> Self {
                Self {
                    inner: Arc::new(inner),
                }
            }
        }

        impl<O> DefaultOptimizerFactory<O> for DynamicOptimizerFactory<O> {
            fn create(&self, param_count: usize, param_dimension: usize) -> O {
                (self.inner)(param_count, param_dimension)
            }
        }

        #[macro_export]
        macro_rules! lazy_opt {
            ($opt:expr, $linear:expr) => {
                $opt.create_lazy_named(format!(
                    "f={},l={},c={}__{}_{}_{}_",
                    file!(),
                    line!(),
                    column!(),
                    stringify!($linear),
                    $linear.stride(),
                    $linear.count()
                ))
            };
            ($opt:expr) => {
                $opt.create_lazy_named(format!("f={},l={},c={}_", file!(), line!(), column!()))
            };
        }

        #[macro_export]
        macro_rules! cached_opt {
            ($opt:expr, $linear:expr) => {
                $opt.create(format!(
                    "f={},l={},c={}__{}_{}_{}_",
                    file!(),
                    line!(),
                    column!(),
                    stringify!($linear),
                    $linear.stride(),
                    $linear.count()
                ))
            };
        }
    }

    pub mod lazy {
        use anyhow::Result;

        use crate::ml::transformer::linear::Linear;

        use super::Optimizer;

        pub struct LazyOptimizer<'a, O> {
            factory: Option<Box<dyn FnOnce(usize, usize) -> O + 'a>>,
            instance: Option<O>,
        }

        impl<'a, O: Optimizer> LazyOptimizer<'a, O> {
            pub fn new<F: FnOnce(usize, usize) -> O + 'a>(factory: F) -> Self {
                Self {
                    factory: Some(Box::new(factory)),
                    instance: None,
                }
            }
        }

        impl<'a, O: Optimizer> Optimizer for LazyOptimizer<'a, O> {
            fn update(&mut self, target: &mut Linear, dloss_dtarget: &Linear) -> Result<()> {
                let instance = self.instance.get_or_insert_with(|| {
                    let param_count = dloss_dtarget.count();
                    let param_dimension = dloss_dtarget.stride();
                    let factory = self.factory.take().unwrap();
                    factory(param_count, param_dimension)
                });

                instance.update(target, dloss_dtarget)
            }
        }
    }

    pub mod builder {
        use super::*;

        pub struct AdamOptimizerBuilder {
            momentum: Linear,
            rms: Linear,
            beta: (NodeValue, NodeValue),
            epsilon: NodeValue,
            eta: NodeValue,
        }

        impl AdamOptimizerBuilder {
            pub fn new(param_count: usize, param_dimension: usize) -> Self {
                Self {
                    momentum: Linear::new(param_count, param_dimension),
                    rms: Linear::new(param_count, param_dimension),
                    beta: (0.9, 0.999),
                    epsilon: 1e-8,
                    eta: 0.001,
                }
            }

            pub fn with_beta(mut self, beta1: NodeValue, beta2: NodeValue) -> Self {
                self.beta = (beta1, beta2);
                self
            }

            pub fn with_epsilon(mut self, epsilon: NodeValue) -> Self {
                self.epsilon = epsilon;
                self
            }

            pub fn with_eta(mut self, eta: NodeValue) -> Self {
                self.eta = eta;
                self
            }

            pub fn build(self) -> AdamOptimizer {
                AdamOptimizer {
                    momentum: self.momentum,
                    rms: self.rms,
                    beta: self.beta,
                    epsilon: self.epsilon,
                    eta: self.eta,
                    t: 1,
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::cell::RefCell;

        use crate::ml::transformer::tests::helpers::{assert_optimisation_converges, new_linear};

        use super::*;

        #[test]
        fn adam_can_optimise_linear() {
            let batch_count = 12;
            let input_dimension = 48;
            let output_dimension = 8;

            let iters = 100;

            let optimizer = AdamOptimizer::new_builder(output_dimension, input_dimension)
                .with_eta(0.1)
                .build();
            let optimizer = RefCell::new(optimizer);

            assert_optimisation_converges(
                &move |rng| {
                    let weights = new_linear(output_dimension, input_dimension, &rng);
                    let inputs = new_linear(batch_count, input_dimension, &rng);
                    let target = new_linear(batch_count, output_dimension, &rng);
                    (weights, inputs, target)
                },
                &move |weights, inputs| inputs.matrix_product_rhs_transposed(&weights),
                &move |weights, inputs, dloss| {
                    let dweights = dloss.matrix_product_lhs_transposed(&inputs);
                    let dinputs = dloss.matrix_product(&weights);
                    optimizer.borrow_mut().update(weights, &dweights).unwrap();
                    dinputs
                },
                iters,
            );
        }

        #[test]
        fn rms_prop_can_optimise_linear() {
            let batch_count = 12;
            let input_dimension = 48;
            let output_dimension = 8;

            let iters = 100;

            let optimizer = RMSpropOptimizer::new(output_dimension, input_dimension)
                .with_eta(0.01)
                .with_gamma(0.999);
            let optimizer = RefCell::new(optimizer);

            assert_optimisation_converges(
                &move |rng| {
                    let weights = new_linear(output_dimension, input_dimension, &rng);
                    let inputs = new_linear(batch_count, input_dimension, &rng);
                    let target = new_linear(batch_count, output_dimension, &rng);
                    (weights, inputs, target)
                },
                &move |weights, inputs| inputs.matrix_product_rhs_transposed(&weights),
                &move |weights, inputs, dloss| {
                    let dweights = dloss.matrix_product_lhs_transposed(&inputs);
                    let dinputs = dloss.matrix_product(&weights);
                    optimizer.borrow_mut().update(weights, &dweights).unwrap();
                    dinputs
                },
                iters,
            );
        }

        #[test]
        fn sgd_can_optimise_linear() {
            let batch_count = 12;
            let input_dimension = 48;
            let output_dimension = 8;

            let iters = 100;
            let learn_rate = 0.01;

            let optimizer = SGDOptimizer::new(learn_rate);
            let optimizer = RefCell::new(optimizer);

            assert_optimisation_converges(
                &move |rng| {
                    let weights = new_linear(output_dimension, input_dimension, &rng);
                    let inputs = new_linear(batch_count, input_dimension, &rng);
                    let target = new_linear(batch_count, output_dimension, &rng);
                    (weights, inputs, target)
                },
                &move |weights, inputs| inputs.matrix_product_rhs_transposed(&weights),
                &move |weights, inputs, dloss| {
                    let dweights = dloss.matrix_product_lhs_transposed(&inputs);
                    let dinputs = dloss.matrix_product(&weights);
                    optimizer.borrow_mut().update(weights, &dweights).unwrap();
                    dinputs
                },
                iters,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    pub mod helpers {
        use crate::ml::{
            layer::LayerInitStrategy, transformer::linear::Linear, NodeValue, RngStrategy,
        };

        pub fn assert_optimisation_converges<T, I>(
            create: &dyn Fn(RngStrategy) -> (T, I, Linear),
            forward: &dyn Fn(&T, &I) -> Linear,
            backward: &dyn Fn(&mut T, &mut I, Linear) -> Linear,
            iters: usize,
        ) {
            let rng = RngStrategy::testable(12345);

            let (mut testable_instance, mut inputs, target) = create(rng.clone());
            let mut iteration = 0;
            let mut initial_mean_loss = None;

            let (output, mean_loss) = loop {
                // compute forward pass
                let output = forward(&testable_instance, &inputs);

                // calculate simple rms loss and gradients compared to target values
                let loss = output.iter().sub(target.iter()).powf_scalar(2.0);
                let dloss = output.iter().sub(target.iter()).multiply_scalar(2.0);

                // compute backward pass, apply gradients, and return inputs gradients
                let grads = backward(&mut testable_instance, &mut inputs, dloss.collect());
                if !grads.is_finite() {
                    println!("{:?}", &output);
                    panic!("failed optimisation: grad value is non-finite");
                }

                // get gradients & loss for reporting
                let abs_grads = grads.iter().abs();
                let mean_grads = abs_grads.flatten_mean().iter_transpose().flatten_mean();
                let mean_grads = mean_grads.as_scalar().unwrap();
                let mean_loss = loss.flatten_mean().iter_transpose().flatten_mean();
                let mean_loss = mean_loss.as_scalar().unwrap();
                initial_mean_loss.get_or_insert(mean_loss);

                // report optimisation iteration
                if (iteration + 1) % (iters / 10).max(1) == 0 || iteration == 0 {
                    println!(
                        "optimisation iter: {:<7} | loss: {:<10.5} | mean abs gradient: {:<5.2}",
                        iteration + 1,
                        mean_loss,
                        mean_grads
                    );
                }

                // progress iteration counter, or complete
                iteration += 1;
                if iteration > iters {
                    break (output, mean_loss);
                }
            };
            match format!(" target: {},\n output: {}", target, output) {
                dump if dump.len() < 1500 => {
                    println!("{dump}")
                }
                _ => {}
            }

            let initial_mean_loss = initial_mean_loss.unwrap();

            assert!(mean_loss.is_finite(), "final loss has invalid value");

            assert!(
                initial_mean_loss > mean_loss,
                "loss failed to optimise (start={:.2}, end={:.2})",
                initial_mean_loss,
                mean_loss
            );
        }

        pub fn assert_input_gradients<T>(
            create: &dyn Fn(RngStrategy) -> (T, Linear, Linear),
            forward: &dyn Fn(&T, &Linear) -> Linear,
            backward: &dyn Fn(&mut T, &Linear, Linear) -> Linear,
        ) {
            let rng = RngStrategy::testable(12345);
            let epsilon = 1e-8;
            let decimals = 4;

            let (mut testable_instance, inputs, target) = create(rng.clone());
            let output = forward(&testable_instance, &inputs);

            let dloss = output.iter().sub(target.iter()).multiply_scalar(2.0);

            let computed_dloss_dinput = backward(&mut testable_instance, &inputs, dloss.collect());
            let expected_dloss_dinput = compute_expected_dloss_dinput(
                |inputs| forward(&testable_instance, inputs),
                |output| {
                    let loss = output.iter().sub(target.iter()).powf_scalar(2.0).collect();
                    loss.to_sum()
                },
                inputs,
                epsilon,
            );

            let delta_computed_dloss_dinput = computed_dloss_dinput
                .iter()
                .sub(expected_dloss_dinput.iter())
                .round(decimals)
                .collect();

            let zeros = Linear::with_dimensions(&delta_computed_dloss_dinput);
            assert_eq!(zeros, delta_computed_dloss_dinput);
        }

        pub fn compute_expected_dloss_dinput(
            forward_fn: impl Fn(&Linear) -> Linear,
            loss_fn: impl Fn(&Linear) -> NodeValue,
            inputs: Linear,
            epsilon: NodeValue,
        ) -> Linear {
            let output = forward_fn(&inputs);
            let original_loss = loss_fn(&output);
            let stride = inputs.stride();

            let inputs_vec: Vec<_> = inputs.values_iter().collect();
            let mut dloss_dinput = vec![];

            for (target_idx, (&source, ..)) in inputs_vec.iter().enumerate() {
                let target = source + epsilon;
                let perturbed = Linear::from_iter(
                    stride,
                    inputs_vec
                        .iter()
                        .enumerate()
                        .map(|(source_idx, (&source, ..))| {
                            if source_idx == target_idx {
                                target
                            } else {
                                source
                            }
                        }),
                );

                let output = forward_fn(&perturbed.unwrap());
                let loss = loss_fn(&output);
                let loss_delta = loss - original_loss;
                let dinput = loss_delta / epsilon;

                dloss_dinput.push(dinput);
            }

            Linear::from_iter(stride, dloss_dinput.into_iter()).unwrap()
        }

        pub fn new_linear(seq_len: usize, embedding_dimension: usize, rng: &RngStrategy) -> Linear {
            let mut linear = Linear::new(seq_len, embedding_dimension);
            linear.initialize_as_layer(&LayerInitStrategy::Kaiming, &rng);
            linear
        }
    }
}
