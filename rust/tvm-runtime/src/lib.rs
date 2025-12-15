mod params;
mod rtensor;

pub use params::*;
pub use rtensor::*;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use tvm_ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType};

    use super::*;

    const RT_FILENAME: &str = "";
    const TENSOR_CACHE_PATH: &str = "";

    #[test]
    fn test_module() -> () {
        let exec = tvm_ffi::Module::load_from_file(RT_FILENAME).unwrap();
        let vm: tvm_ffi::Module = exec
            .get_function("vm_load_executable")
            .unwrap()
            .call_tuple(())
            .unwrap()
            .try_into()
            .unwrap();
        vm.get_function("vm_initialization")
            .unwrap()
            .call_tuple((
                tvm_ffi::DLDeviceType::kDLMetal as i32, // device_type
                0 as i32,                               // device_id
                2i32,                                   // vm_allocator_type
                tvm_ffi::DLDeviceType::kDLCPU as i32,   // host_device_type
                0i32,                                   // host_device_id
                2i32,                                   // host_vm_allocator_type
            ))
            .unwrap();

        let tensor_cache_path = PathBuf::from(TENSOR_CACHE_PATH);
        let tensor_cache =
            TensorCache::from(&tensor_cache_path, DLDeviceType::kDLMetal, 0).unwrap();
        println!("{:?}", tensor_cache);
    }

    #[test]
    fn test_tensor() -> () {
        let mut tensor = RTensor::new(
            DLDevice {
                device_type: tvm_ffi::DLDeviceType::kDLMetal,
                device_id: 0,
            },
            [3, 3],
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
        );
        let mut src = Vec::<f32>::new();
        for _ in 0..9 {
            src.push(0.);
        }
        let src_u8: &[u8] = unsafe {
            core::slice::from_raw_parts(
                src.as_ptr() as *const u8,
                src.len() * core::mem::size_of::<f32>(),
            )
        };
        tensor.copy_from(src_u8);
    }
}
