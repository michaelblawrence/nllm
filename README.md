# nllm
Train and prompt (**n**ot so) **l**arge **l**anguage **m**odels, powered by a transformer architecture. Built for educational purposes.

[Try ChatNLLM! It is powered by `nllm` to create a chat room with AI!](https://notlargelanguagemodel.com)

## Features

- Train models from the CLI from any plaintext file
- Integrates with AWS lambda for cloud-based inference, or get started with CLI chat-based tool 'respond'
- Supports multiple model architectures, including full or decoder-only transformers (GPT support is coming soon)
- Works on my machine! Truly a 'work in progress', so check it out, but just not in production!

## Why `nllm`?

With the major developments around machine learning and natural language processing seemingly happening around the space of "Python APIs, C++ under-the-hood, blazing fast GPUs, 3.14159 bajillion tokens...", its good to take a step back and see what can be done with less.

Its almost like the development of not large language models (or 'NLLMs') seems almost inevitable.

The aim of this project is to demystify the concepts used in creating larger language models (and their chat-bot interfaces) for the purpose of education.
For this reason this crate will often avoid pulling in additional dependencies around machine learning and math, in an effort to provide solutions to each obstacle from first-principles. 

## Installation

Pull the latest code from 'master' branch and run the following in the terminal

```sh
cargo install --path ./nllm_embed
```

## Quick Start

### Training models from scratch
To train your first model, run the following command:

```sh
embed --use-gdt -i ./res/names.txt -O my-first-model
```

This will start up the nllm model trainer (called 'embed').
The trainer will read in the provided txt file and create a tokenizer (default is character-level tokens).
A generative decoder-only transformer (or 'GDT') model will be initiated and the training will begin.

At any time during the training pressing the `H` key and then the `Enter` key will display a help menu for usable commands.
Once the training has reached the max iterations the model will be exported to disk (in this case to "out/my-first-model.json").

### Resuming from checkpoint
You can resume training from an exported model checkpoint by running the following command:

```sh
embed load out/my-first-model.json
```

When resuming another input plaintext file can be provided to fine-tune the model:

```sh
embed load custom-base-models/en_gb.json -O fine-tuned-model -i ./res/tinyshakespeare.txt
```

### Generating from model
To generate from your model, run the following command:

```sh
respond out/my-first-model.json
```

This will start up the nllm model inference CLI (called 'respond').
The respond CLI will provide the user to input a starting prompt from which the nllm model will continue to generate text.
The prompt can also be left blank for the model to randomly sample the all tokens. 

## Note on Crate Organization

`nllm` is divided into two separate crates to organize the underlying engine library code and related binaries:

1. [`nllm_plane`](./nllm_plane):
   This is the main library that offers the nllm engine that powers the model at training and inference time.
2. [`nllm_embed`](./nllm_embed):
   This is the command-line trainer binary code, designed to produce model snapshots that can later be used for inference/generation.
2. [`nllm_respond`](./nllm_respond):
   This is the command-line model inference binary code, designed to consume nllm model snapshots for token inference evaluation with user provided prompts.
2. [`nllm_chat_web`](./nllm_chat_web):
   This is the web API used to power [ChatNLLM](https://notlargelanguagemodel.com), providing real-time messaging and handling of model inference requests over AWS lambda.
2. [`nllm_chat_functions`](./nllm_chat_functions):
   This is the AWS lambda function used to run inference of nllm models for the `nllm_chat_web` API instances.

## Contributing

If you're interested in contributing to `nllm`, please follow standard Rust community guidelines and submit a PR on our repository.

## License

This project is dual licensed under [MIT License](./LICENSE-MIT) and [Apache License](./LICENSE-APACHE).
