use core::{ffi::c_void, ptr::null_mut};

use tvm_ffi::{
    collections::tensor::DLTensorExt as _, DLDataType, DLDevice, DLDeviceType, NDAllocator,
};
use tvm_ffi_sys::DLTensor;
use tvm_runtime_sys::{TVMDeviceAPIAllocDataSpace, TVMDeviceAPIFreeDataSpace, TVMDeviceAPIGet};

pub struct DeviceNDAlloc {}

unsafe impl Send for DeviceNDAlloc {}

unsafe impl Sync for DeviceNDAlloc {}

unsafe impl NDAllocator for DeviceNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut c_void {
        let numel = prototype.numel() as usize;
        let item_size = prototype.item_size();
        let size = numel * item_size as usize;
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        TVMDeviceAPIAllocDataSpace(
            TVMDeviceAPIGet(prototype.device, false as i32),
            prototype.device,
            layout.size(),
            layout.align(),
            prototype.dtype,
        )
    }

    unsafe fn free_data(&mut self, tensor: &DLTensor) {
        TVMDeviceAPIFreeDataSpace(
            TVMDeviceAPIGet(tensor.device, false as i32),
            tensor.device,
            tensor.data,
        );
    }
}

pub struct RTensor {
    shape: Vec<i64>,
    dltensor: DLTensor,
}

unsafe impl Send for RTensor {}

impl RTensor {
    pub fn new(
        device: DLDevice,
        ndim: i32,
        dtype: DLDataType,
        shape: impl IntoIterator<Item = impl Into<i64>>,
        byte_offset: u64,
    ) -> Self {
        let mut shape = shape.into_iter().map(|v| v.into()).collect::<Vec<_>>();
        let mut dltensor = DLTensor {
            data: null_mut(),
            device,
            ndim,
            dtype,
            shape: shape.as_mut_ptr(),
            strides: null_mut(),
            byte_offset,
        };
        dltensor.data = unsafe { DeviceNDAlloc {}.alloc_data(&dltensor) };
        Self { shape, dltensor }
    }

    pub fn host_device_type(&self) -> Option<DLDeviceType> {
        match self.dltensor.device.device_type {
            DLDeviceType::kDLCPU => None,
            DLDeviceType::kDLCUDA => Some(DLDeviceType::kDLCUDAHost),
            DLDeviceType::kDLCUDAHost => None,
            DLDeviceType::kDLOpenCL => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLVulkan => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLMetal => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLVPI => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLROCM => Some(DLDeviceType::kDLROCMHost),
            DLDeviceType::kDLROCMHost => None,
            DLDeviceType::kDLExtDev => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLCUDAManaged => Some(DLDeviceType::kDLCUDAHost),
            DLDeviceType::kDLOneAPI => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLWebGPU => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLHexagon => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLMAIA => Some(DLDeviceType::kDLCPU),
            DLDeviceType::kDLTrn => Some(DLDeviceType::kDLCPU),
        }
    }

    pub fn is_host_tensor(&self) -> bool {
        !self.is_device_tensor()
    }

    pub fn is_device_tensor(&self) -> bool {
        self.host_device_type().is_some()
    }

    pub fn copy_from(&mut self, data: &[u8]) -> () {
        if self.is_host_tensor() {
            let numel = self.dltensor.numel() as usize;
            let item_size = self.dltensor.item_size() as usize;
            let expected_bytes = numel * item_size;
            debug_assert_eq!(
                expected_bytes,
                data.len(),
                "RTensor::copy_data: data length ({}) does not match tensor size ({})",
                data.len(),
                expected_bytes
            );
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    self.dltensor.data as *mut u8,
                    expected_bytes,
                )
            };
            return;
        }

        if let Some(host_device_type) = self.host_device_type() {
            let mut host_dltensor = DLTensor {
                data: data.as_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: host_device_type,
                    device_id: 0,
                },
                ndim: self.dltensor.ndim,
                dtype: self.dltensor.dtype,
                shape: self.dltensor.shape,
                strides: self.dltensor.strides,
                byte_offset: self.dltensor.byte_offset,
            };

            unsafe {
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    TVMDeviceAPIGet(self.dltensor.device, false as i32),
                    &mut host_dltensor,
                    &mut self.dltensor,
                    null_mut(),
                )
            };
        }
    }
}

impl Drop for RTensor {
    fn drop(&mut self) {
        unsafe { DeviceNDAlloc {}.free_data(&self.dltensor) };
    }
}

impl AsRef<DLTensor> for RTensor {
    fn as_ref(&self) -> &DLTensor {
        &self.dltensor
    }
}

impl AsMut<DLTensor> for RTensor {
    fn as_mut(&mut self) -> &mut DLTensor {
        &mut self.dltensor
    }
}
