fn main() {
    // libtorch_cuda.so registers the CUDA backend via a static initializer.
    // The default --as-needed linker flag drops it because no symbols are
    // directly referenced from Rust code. Re-link it with --no-as-needed
    // so the linker keeps the library and the initializer runs at load time.
    #[cfg(feature = "nn")]
    {
        println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed,-ltorch_cuda,--as-needed");
    }
}
