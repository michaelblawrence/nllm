set dotenv-load

model_output_label := "dev-model"
ssh_user := env_var_or_default('SSH_USER', 'root')
home_dir := env_var_or_default('USER_HOME_DIR', '/home/docker')

default:
  @just --list

build:
  cargo build --features="multi_threaded" --release --package embed

build-wasi:
  cargo +nightly wasi build --no-default-features --features="wasi" --release --package embed 

docker-build:
  docker build -t embed .

install:
  cargo install --features="multi_threaded" --offline --path ./nllm_embed

cargo-run input_file="./res/tinyshakespeare.txt" rounds="100000" output_label=(model_output_label):
  cargo run --features="multi_threaded" --release --package embed -- \
   --phrase-test-set-max-tokens 500 --use-gdt --repl "PR" --training-rounds {{rounds}}
  -i {{input_file}} --output-label-append-details -o out/labelled/train -O {{model_output_label}}

cargo-resume input_file repl="p" batch_size="32":
  cargo run --features="multi_threaded" --release --package embed -- \
    load {{input_file}} \
    --repl "{{repl}}" --batch-size {{batch_size}} --force-continue

cargo-respond input_file:
  cargo run --features="threadrng" --release --package respond -- \
    {{input_file}}

cargo-flamegraph input_file:
  sudo cargo flamegraph --no-default-features --features="single_threaded thread" --package embed -- \
    load {{input_file}} \
    --repl="pr"

run input_file="./res/tinyshakespeare.txt" rounds="100000":
  embed --quit-on-complete --phrase-test-set-max-tokens 500 --use-gdt --repl "PR" \
    --training-rounds {{rounds}} -i {{input_file}} -o out -O {{model_output_label}}

run-microgpt:
  embed --quit-on-complete --phrase-test-set-max-tokens 2500 --use-gdt -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt

resume input_file repl="p" batch_size="32":
  embed load {{input_file}} \
    --repl "{{repl}}" --batch-size {{batch_size}} --force-continue

serve-chat-model input_file:
  NLLM_MODEL_PATH={{input_file}} NLLM_TRIPLE_HASH_PROMPT=1 NLLM_INFER_TIMEOUT_MS=30000 \
   cargo watch -x "run --release --package nllm_chat_web"

deploy-chat-model input_file alias="beta": (build-lambda) (deploy-s3 input_file) (deploy-lambda file_name(input_file) alias)

deploy-s3 input_file:
  aws s3 cp {{input_file}} s3://$NLLM_MODEL_S3_BUCKET

deploy-lambda model_s3_key alias="beta" version_tag="v4":
  cargo lambda deploy --tags project=nllm_chat_infer,cversion={{version_tag}} \
    --env-var NLLM_MODEL_S3_BUCKET=$NLLM_MODEL_S3_BUCKET --env-var NLLM_MODEL_S3_KEY={{model_s3_key}} --env-var NLLM_TRIPLE_HASH_PROMPT=1 --env-var NLLM_TEMP=$NLLM_TEMP \
    --role $NLLM_LAMBDA_ARN --timeout $NLLM_LAMBDA_TIMEOUT --memory $NLLM_LAMBDA_MEMORY --enable-function-url --alias {{alias}} \
    nllm_chat_functions

build-lambda:
  cargo lambda build --release --arm64 --package nllm_chat_functions

docker-run run_script="run" script_args="": docker-build
  mkdir -p out/script-{{run_script}}
  docker run --name embedtestrun --rm -itd -v `pwd`/out/script-{{run_script}}:/app/out \
    embed just {{run_script}} {{script_args}}
  docker attach --no-stdin --sig-proxy=false embedtestrun

wasi-run input_file="./res/tinyshakespeare.txt" rounds="100000" output_label=(model_output_label):
  wasmer run --dir ./out --dir ./res --enable-all ./target/wasm32-wasi/release/embed.wasi.wasm -- \
    --quit-on-complete --phrase-test-set-max-tokens 5 --use-gdt \
    --training-rounds {{rounds}} -i {{input_file}} -o out -O {{model_output_label}}

run-like-wasi-testrun:
  cargo run --no-default-features --features="single_threaded" --release --package embed -- \
    --quit-on-complete --single-batch-iterations --char --training-rounds 100000 \
    --batch-size 64 --phrase-test-set-max-tokens 5 \
    -i ./res/tinyshakespeare.txt -o out -O dev-model-wasi
