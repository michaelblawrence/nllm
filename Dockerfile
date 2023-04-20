FROM rust:1.68.2-alpine3.17

RUN set -eux; \
        apk add --no-cache musl-dev; \
		rustup --version; \
        cargo install just;

WORKDIR /app
RUN set -eux; \
		cargo init --vcs none --bin; \
		ls -l; \
		mkdir src/bin; \
		mkdir src/bin/embed; \
		cp src/main.rs src/bin/derivate.rs; \
		cp src/main.rs src/bin/embed/main.rs;

COPY Cargo.toml Cargo.lock ./
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
        
RUN set -eux; \
		cargo build --features="cli thread threadpool" --release --bin embed;

COPY src src
COPY justfile ./

RUN set -eux; \
        touch src/bin/embed/main.rs src/bin/derivate.rs src/main.rs; \
		just build; \
        just install;

COPY res/tinyimdb.txt ./
COPY res/tinyshakespeare.txt ./