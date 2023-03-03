py:
	maturin develop

build:
	cargo build

release:
	cargo build --release

test:
	cargo test

clean:
	rm -rf target/

