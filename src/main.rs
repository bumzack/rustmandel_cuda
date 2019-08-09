#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rustacuda::memory::DeviceBox;

fn sum() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_kernel2/81b77159abf25dc3/nvptx64-nvidia-cuda/release/cuda_kernel2.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate space on the device and copy numbers to it.
    let mut x = DeviceBuffer::from_slice(&[10.0f32, 20.0, 30.0, 40.0])?;
    let mut y = DeviceBuffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0])?;
    let mut result = DeviceBuffer::from_slice(&[0.0f32, 0.0, 0.0, 0.0])?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        launch!(module.add<<<1, 4, 0, stream>>>(
            x.as_device_ptr(),
            y.as_device_ptr(),
            result.as_device_ptr(),
            result.len()
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host = [0.0f32, 0.0, 0.0, 0.0];
    result.copy_to(&mut result_host)?;

    println!("bla sum is {:?}", result_host);

    Ok(())
}



fn other_kernel() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_kernel2/81b77159abf25dc3/nvptx64-nvidia-cuda/release/cuda_kernel2.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate space on the device and copy numbers to it.
    let mut x = DeviceBox::new(&2.0f32).unwrap();
    let mut y = DeviceBox::new(&3.0f32).unwrap();
    let mut result = DeviceBox::new(&0.0f32).unwrap();

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        launch!(module.example_kernel<<<4, 4, 4, stream>>>(
            x.as_device_ptr(),
            y.as_device_ptr()
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host = 0.0f32;
    result.copy_to(&mut result_host)?;

    println!("xxxx  Sum is {:?}", result_host);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // other_kernel()?;
    sum()
}