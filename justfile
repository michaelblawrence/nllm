mongodb_user := env_var_or_default('MONGO_INITDB_ROOT_USERNAME', 'root')
mongodb_pass := env_var_or_default('MONGO_INITDB_ROOT_PASSWORD', 'example')
mongodb_host := env_var_or_default('MONGO_INITDB_HOST', 'localhost')
model_output_label := "dev-model"
ssh_user := env_var_or_default('SSH_USER', 'root')
home_dir := env_var_or_default('USER_HOME_DIR', '/home/docker')

build:
  cargo build --features="multi_threaded" --release --bin embed

build-wasi:
  cargo +nightly wasi build --features="wasi" --release --bin embed 

install:
  cargo install --features="multi_threaded" --bin embed --offline --path .

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

run-microgpt:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 24 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 6,4  --embedding-size 128 --input-stride-width 256 --repl "PR" \
    --training-rounds 50000 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt \
    --use-transformer

run-microgpt-MK2:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 8 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 2,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 1000 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt \
    --use-transformer

run-microgpt-MK3:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 64 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 3,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 1000 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt \
    --use-transformer

run-microgpt-MK4:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 96 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 3,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 10 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt \
    --use-transformer

run-microgpt-MK5:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 384 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 3,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 10 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgpt \
    --use-transformer

run-microgpt-MK6:
  embed json --trainer-config '{"activation_mode":"Tanh","batch_size":64,"embedding_size":128,"hidden_deep_layer_nodes":"3,4","hidden_layer_nodes":512,"input_stride_width":256,"input_txt_path":"./res/tinyimdbtrainneg.txt","output_dir":"out/labelled/train","output_label":"microgpt-e128-L512x3x4","output_label_append_details":false,"pause_on_start":false,"phrase_split_seed":null,"phrase_test_set_max_tokens":500,"phrase_test_set_split_pct":20.0,"phrase_train_set_size":null,"phrase_word_length_bounds":[null,null],"quit_on_complete":false,"repl":"crP","sample_from_newline":false,"single_batch_iterations":true,"snapshot_interval_secs":120,"train_rate":0.004,"training_rounds":1000,"use_character_tokens":true,"use_transformer":true}'

run-microgpt-MK7 phrase_test_set_max_tokens="500":
  embed json --trainer-config '{"activation_mode":"Tanh","batch_size":8,"embedding_size":128,"hidden_deep_layer_nodes":"1,4","hidden_layer_nodes":512,"input_stride_width":128,"input_txt_path":"./res/tinyimdbtrainneg.txt","output_dir":"out/labelled/train","output_label":"microgpt","output_label_append_details":true,"pause_on_start":false,"phrase_split_seed":null,"phrase_test_set_max_tokens":{{phrase_test_set_max_tokens}},"phrase_test_set_split_pct":20.0,"phrase_train_set_size":null,"phrase_word_length_bounds":[null,null],"quit_on_complete":false,"repl":"crP","sample_from_newline":false,"single_batch_iterations":true,"snapshot_interval_secs":120,"train_rate":0.004,"training_rounds":1000,"use_character_tokens":true,"use_transformer":true}'

run-microgpt-MK8:
  cargo run --features="multi_threaded" --release --bin embed -- \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 1 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 2,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 100 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgdt-dev-init \
    --use-transformer --use-gdt --gdt-word-mode --gdt-bpe-vocab-size 2500

run-microgpt-MK9:
  embed \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 24 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 2,4  --embedding-size 128 --input-stride-width 256 --repl "P" \
    --training-rounds 100 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgdt-dev-init \
    --use-transformer --use-gdt --gdt-word-mode --gdt-bpe-vocab-size 2500

run-microgpt-MK10:
  embed \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 72 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 512 -H 6,4  --embedding-size 128 --input-stride-width 512 --repl "P" \
    --training-rounds 100 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgdt-dev-init \
    --use-transformer --use-gdt --gdt-word-mode --gdt-bpe-vocab-size 2500

run-microgpt-MK11:
  cargo run --features="multi_threaded" --release --bin embed -- \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 4 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 768 -H 1,6  --embedding-size 192 --input-stride-width 256 --repl "P" \
    --training-rounds 100 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgdt-dev-init \
    --use-transformer --use-gdt --gdt-word-mode --gdt-bpe-vocab-size 2500

run-microgpt-MK12:
  cargo run --features="multi_threaded" --release --bin embed -- \
    --single-batch-iterations --char --train-rate 0.004 --batch-size 4 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 72 -H 1,4  --embedding-size 16 --input-stride-width 256 --repl "P/" \
    --training-rounds 100 -i ./res/tinyimdbtrainneg.txt \
    --output-label-append-details -o out/labelled/train -O microgdt-dev-init \
    --use-transformer --use-gdt --gdt-bpe-enable --gdt-bpe-vocab-size 2500

upload-tinyshakespeare mongodb_conn_str=("mongodb://" + mongodb_user + ":" + mongodb_pass + "@" + mongodb_host + ":27017"):
  MONGODB_HOST={{mongodb_conn_str}} \
    upload out/{{model_output_label}}

upload mongodb_conn_str=("mongodb://" + mongodb_user + ":" + mongodb_pass + "@" + mongodb_host + ":27017"):
  MONGODB_HOST={{mongodb_conn_str}} \
    upload out/{{model_output_label}}

docker-build:
  docker build -t embed .

docker-build-aws:
  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 545568271585.dkr.ecr.us-east-1.amazonaws.com
  docker pull 545568271585.dkr.ecr.us-east-1.amazonaws.com/embed-ecr:latest
  docker tag 545568271585.dkr.ecr.us-east-1.amazonaws.com/embed-ecr:latest embed:0.1
  docker build -t embed:0.2 .
  docker tag embed:0.2 545568271585.dkr.ecr.us-east-1.amazonaws.com/embed-ecr:latest
  docker push 545568271585.dkr.ecr.us-east-1.amazonaws.com/embed-ecr:latest

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
  docker run --name embedtestrun --rm -itd -v `pwd`/out/script-{{run_script}}:/app/out \
    embed just {{run_script}} {{script_args}}
  docker attach --no-stdin --sig-proxy=false embedtestrun

docker-run-aws run_script="run" script_args="": docker-build-aws
  mkdir -p out/script-{{run_script}}
  docker run --name embedtestrun --rm -itd -v `pwd`/out/script-{{run_script}}:/app/out \
    545568271585.dkr.ecr.us-east-1.amazonaws.com/embed-ecr:latest just {{run_script}} {{script_args}}
  docker attach --no-stdin --sig-proxy=true embedtestrun

configure-node node_ip public_key_path:
  ssh-copy-id -i {{public_key_path}} {{ssh_user}}@{{node_ip}}
  ssh {{ssh_user}}@{{node_ip}} snap install --edge --classic just
  ssh {{ssh_user}}@{{node_ip}} apt install -y socat

configure-aws-arm-ami-node node_ip public_key_path:
  ssh ec2-user@{{node_ip}} "sudo yum update -y; sudo yum search docker; sudo yum install docker -y; sudo usermod -a -G docker ec2-user; id ec2-user; newgrp docker; sudo systemctl enable docker.service; sudo systemctl start docker.service; sudo yum install socat -y; sudo yum install gcc -y; curl https://sh.rustup.rs -sSf > RUSTUP.sh; sh RUSTUP.sh -y; source ~/.bashrc; cargo install just"

# m5zn.3xlarge
configure-aws-docker-node node_ip:
  ssh {{ssh_user}}@{{node_ip}} sudo snap install --edge --classic just
  ssh {{ssh_user}}@{{node_ip}} sudo snap install --edge --classic aws-cli
  ssh {{ssh_user}}@{{node_ip}} sudo apt install -y socat
  ssh {{ssh_user}}@{{node_ip}} aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 545568271585.dkr.ecr.us-east-1.amazonaws.com

configure-aws-arm-docker-node node_ip:
  ssh {{ssh_user}}@{{node_ip}} mkdir -p ~/bin
  ssh {{ssh_user}}@{{node_ip}} curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
  ssh {{ssh_user}}@{{node_ip}} export PATH="$PATH:$HOME/bin"
  ssh {{ssh_user}}@{{node_ip}} sudo snap install --edge --classic aws-cli
  ssh {{ssh_user}}@{{node_ip}} sudo apt install -y socat
  ssh {{ssh_user}}@{{node_ip}} aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 545568271585.dkr.ecr.us-east-1.amazonaws.com

resume-node node_ip input_file version="v1" run_script="testrun-tinyshakespeare":
  ssh {{ssh_user}}@{{node_ip}} mkdir -p {{home_dir}}/code/{{version}}
  git archive --add-file=justfile --add-file=Dockerfile --format=zip HEAD > ./res/archive.zip
  scp -p ./res/archive.zip {{ssh_user}}@{{node_ip}}:{{home_dir}}/code
  ssh {{ssh_user}}@{{node_ip}} "cd {{home_dir}}/code/{{version}}; unzip -o ../archive.zip"
  scp -p ./res/tiny*.txt {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/res/
  scp -p ./{{input_file}} {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/{{input_file}}
  ssh {{ssh_user}}@{{node_ip}} "cd {{home_dir}}/code/{{version}}; just docker-run resume {{input_file}}"

push-node node_ip version="v1" run_script="testrun-tinyshakespeare" script_args="" recipe="docker-run":
  ssh {{ssh_user}}@{{node_ip}} mkdir -p {{home_dir}}/code/{{version}}
  git archive --add-file=justfile --add-file=Dockerfile --format=zip HEAD > ./res/archive.zip
  scp -p ./res/archive.zip {{ssh_user}}@{{node_ip}}:{{home_dir}}/code
  ssh {{ssh_user}}@{{node_ip}} "cd {{home_dir}}/code/{{version}}; unzip -o ../archive.zip"
  scp -p ./res/tiny*.txt {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/res/
  ssh {{ssh_user}}@{{node_ip}} "cd {{home_dir}}/code/{{version}}; just {{recipe}} {{run_script}} {{script_args}}"
  mkdir -p out/remote/{{run_script}}/{{version}}
  scp -p -r {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/out/script-{{run_script}} ./out/remote/{{run_script}}/{{version}}

snapshot-model-node node_ip version="v1":
  ssh {{ssh_user}}@{{node_ip}} "cd {{home_dir}}/code/{{version}}; echo 's' | socat EXEC:'docker attach embedtestrun',pty STDIN"

repl-model-node node_ip repl:
  ssh {{ssh_user}}@{{node_ip}} "echo '{{repl}}' | socat EXEC:'docker attach embedtestrun',pty STDIN"

docker-stop-node node_ip:
  ssh {{ssh_user}}@{{node_ip}} docker kill embedtestrun

push-node-res node_ip version res_fname:
  ssh {{ssh_user}}@{{node_ip}} mkdir -p {{home_dir}}/code/{{version}}
  scp -p ./res/{{res_fname}} {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/res/

clean-node node_ip version="v1" run_script="testrun-tinyshakespeare":
  ssh {{ssh_user}}@{{node_ip}} rm -rf {{home_dir}}/code/{{version}}/out/script-{{run_script}}

pull-node node_ip version="v1" run_script="testrun-tinyshakespeare":
  mkdir -p out/remote/{{run_script}}/{{version}}
  scp -p -r {{ssh_user}}@{{node_ip}}:{{home_dir}}/code/{{version}}/out/script-{{run_script}} ./out/remote/{{run_script}}/{{version}}

pull-aws-node node_ip version="v1" label="testrun-tinyshakespeare":
  mkdir -p out/remote/{{label}}/aws-{{version}}
  scp -p -r {{ssh_user}}@{{node_ip}}:/home/ubuntu/code/out/{{label}} ./out/remote/{{label}}/aws-{{version}}
