mongodb_user := env_var_or_default('MONGO_INITDB_ROOT_USERNAME', 'root')
mongodb_pass := env_var_or_default('MONGO_INITDB_ROOT_PASSWORD', 'example')
mongodb_host := env_var_or_default('MONGO_INITDB_HOST', 'localhost')
model_output_label := "dev-model"

build:
  cargo build --features="multi_threaded" --release --bin embed

build-wasi:
  cargo +nightly wasi build --features="wasi" --release --bin embed 

install:
  cargo install --features="multi_threaded" --bin embed --path .

install-upload:
  cargo install --features="db" --bin upload --path .

wasi-run:
  wasmer run --dir ./out --dir ./res --enable-all ./embed.wasm -- \
    --quit-on-complete --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 \
    --training-rounds 1000 -i ./res/imdb-train-pos.txt -o out -O imdb-model-pos-wasi

wasi-testrun:
  wasmer run --dir ./out --dir ./res --enable-all ./target/wasm32-wasi/release/embed.wasi.wasm -- \
    --quit-on-complete --single-batch-iterations --char --training-rounds 100000 \
    --batch-size 64 --phrase-test-set-max-tokens 5 \
    -i ./res/tinyshakespeare.txt -o out -O dev-model-wasi 

run-like-wasi-testrun:
  cargo run --features="single_threaded" --release --bin embed -- \
    --quit-on-complete --single-batch-iterations --char --training-rounds 100000 \
    --batch-size 64 --phrase-test-set-max-tokens 5 \
    -i ./res/tinyshakespeare.txt -o out -O dev-model-wasi 

cargo-run input_file="./res/tinyshakespeare.txt" rounds="100000" output_label=(model_output_label):
  cargo run --features="multi_threaded" --release --bin embed -- \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    --training-rounds {{rounds}} -i {{input_file}} \
    --output-label-append-details -o out/labelled/train -O {{model_output_label}}

cargo-resume input_file repl="p" batch_size="32":
  cargo run --features="multi_threaded" --release --bin embed -- \
    load {{input_file}} \
    --repl "{{repl}}" --batch-size {{batch_size}} --force-continue

cargo-respond input_file:
  cargo run --features="threadrng" --release --bin respond -- \
    {{input_file}}

cargo-flamegraph input_file:
  sudo cargo flamegraph --features="single_threaded thread" --bin=embed -- \
    load out/labelled/train/dev-model-qa-e24-L64x1/model-r24957-41pct.json \
    --repl="pr"

run input_file="./res/tinyshakespeare.txt" rounds="100000":
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 256 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    --training-rounds {{rounds}} -i {{input_file}} \
    -o out -O {{model_output_label}}

resume input_file repl="p" batch_size="32":
  embed load {{input_file}} \
    --repl "{{repl}}" --batch-size {{batch_size}} --force-continue

testrun-heavy:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0010 --batch-size 256 \
    --phrase-word-length-bounds .. --training-rounds 15000 --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "" \
    -o out -O {{model_output_label}} -i ./res/tinyshakespeare.txt

testrun-tinyshakespeare:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --training-rounds 15 --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    -o out -O {{model_output_label}} -i ./res/tinyshakespeare.txt

upload-tinyshakespeare mongodb_conn_str=("mongodb://" + mongodb_user + ":" + mongodb_pass + "@" + mongodb_host + ":27017"):
  MONGODB_HOST={{mongodb_conn_str}} \
    upload out/{{model_output_label}}

upload mongodb_conn_str=("mongodb://" + mongodb_user + ":" + mongodb_pass + "@" + mongodb_host + ":27017"):
  MONGODB_HOST={{mongodb_conn_str}} \
    upload out/{{model_output_label}}

docker-build:
  docker build -t embed .

docker-down-mongo:
  docker rm -f mongo mongo-express || true

docker-up-mongo: docker-down-mongo
  docker run -d \
    --name mongo \
    --restart always \
    -p 27017:27017 \
    -e MONGO_INITDB_ROOT_USERNAME={{mongodb_user}} \
    -e MONGO_INITDB_ROOT_PASSWORD={{mongodb_pass}} \
    mongo
  docker run -d \
    --name mongo-express \
    --restart always \
    --network host \
    -e ME_CONFIG_MONGODB_ADMINUSERNAME={{mongodb_user}} \
    -e ME_CONFIG_MONGODB_ADMINPASSWORD={{mongodb_pass}} \
    -e ME_CONFIG_MONGODB_URL=mongodb://{{mongodb_user}}:{{mongodb_pass}}@localhost:27017 \
    mongo-express

docker-test: docker-build
  docker run --name embedtestupload --rm embed just testrun-tinyshakespeare

docker-test-tty: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare

docker-test-upload: docker-build
  docker run --name embedtestupload --rm embed just testrun-tinyshakespeare upload-tinyshakespeare

docker-test-upload-tty: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare upload-tinyshakespeare

docker-run run_script="run" script_args="": docker-build
  mkdir -p out/script-{{run_script}}
  docker run --name embedtestrun --rm -v `pwd`/out/script-{{run_script}}:/app/out \
    embed just {{run_script}} {{script_args}}

configure-node node_ip public_key_path:
  ssh-copy-id -i {{public_key_path}} root@{{node_ip}}
  ssh root@{{node_ip}} snap install --edge --classic just

push-node node_ip version="v1" run_script="testrun-tinyshakespeare" script_args="":
  ssh root@{{node_ip}} mkdir -p /home/docker/code/{{version}}
  git archive --add-file=justfile --add-file=Dockerfile --format=zip HEAD > ./res/archive.zip
  scp -p ./res/archive.zip root@{{node_ip}}:/home/docker/code
  ssh root@{{node_ip}} "cd /home/docker/code/{{version}}; unzip -o ../archive.zip"
  scp -p ./res/tiny*.txt root@{{node_ip}}:/home/docker/code/{{version}}/res/
  ssh root@{{node_ip}} "cd /home/docker/code/{{version}}; just docker-run {{run_script}} {{script_args}}"

pull-node node_ip version="v1" run_script="testrun-tinyshakespeare" script_args="":
  mkdir -p out/remote/{{run_script}}/{{version}}
  scp -p root@{{node_ip}}:/home/docker/code/{{version}}/out/script-{{run_script}} ./out/remote/{{run_script}}/{{version}}

pull-aws-node node_ip version="v1" label="testrun-tinyshakespeare" script_args="":
  mkdir -p out/remote/{{label}}/aws-{{version}}
  scp -p -r ubuntu@{{node_ip}}:/home/ubuntu/code/out/{{label}} ./out/remote/{{label}}/aws-{{version}}
