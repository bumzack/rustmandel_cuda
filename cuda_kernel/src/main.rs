#![cfg_attr(target_os = "cuda", feature(abi_ptx, proc_macro_hygiene))]
#![cfg_attr(target_os = "cuda", no_std)]

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn example_kernel(a: f64, b: f64) {
    use ptx_support::prelude::*;

    cuda_printf!(
        "Hello from block(%lu,%lu,%lu) and thread(%lu,%lu,%lu)\n",
        Context::block().index().x,
        Context::block().index().y,
        Context::block().index().z,
        Context::thread().index().x,
        Context::thread().index().y,
        Context::thread().index().z,
    );

    if Context::block().index() == (0, 0, 0) && Context::thread().index() == (0, 0, 0) {
        cuda_printf!("\n");
        cuda_printf!("extra formatting:\n");
        cuda_printf!("int(%f + %f) = int(%f) = %d\n", a, b, a + b, (a + b) as i32);
        cuda_printf!("ptr(\"%s\") = %p\n", "first", "first".as_ptr());
        cuda_printf!("ptr(\"%s\") = %p\n", "other", "other".as_ptr());
    }
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn  add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    // let i = accel_core::index();
    use ptx_support::prelude::*;
    let i  = Context::thread().index().x as isize;

    cuda_printf!(
        "ADD:  block(%lu,%lu,%lu) and thread(%lu,%lu,%lu)\n",
        Context::block().index().x,
        Context::block().index().y,
        Context::block().index().z,
        Context::thread().index().x,
        Context::thread().index().y,
        Context::thread().index().z,
    );
    cuda_printf!("ADD:  n = %li,  i = %li   \n", n as i64, i as i64);

    if (i as usize) < n {
        cuda_printf!("ADD:   i = %li     a[i] = %f,  b[i] = %f,    \n",   i as i64, *a.offset(i) as f64, *b.offset(i) as f64);
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}


fn main() {}
