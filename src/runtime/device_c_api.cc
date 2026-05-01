#include <tvm/runtime/device_api.h>

using tvm::Device;
using tvm::runtime::DeviceAPI;

extern "C" {
struct TVMDeviceAPI {
  DeviceAPI* impl;
};

TVM_DLL TVMDeviceAPI* TVMDeviceAPIGet(DLDevice dev, bool allow_missing) {
  DeviceAPI* api = DeviceAPI::Get(dev, allow_missing);
  if (!api) {
    return nullptr;
  }
  TVMDeviceAPI* handle = new TVMDeviceAPI();
  handle->impl = api;
  return handle;
}

TVM_DLL void TVMDeviceAPIDestroy(TVMDeviceAPI* handle) { delete handle; }

TVM_DLL void TVMDeviceAPISetDevice(TVMDeviceAPI* handle, DLDevice dev) {
  handle->impl->SetDevice(dev);
}

TVM_DLL void TVMDeviceAPIGetAttr(TVMDeviceAPI* handle, DLDevice dev, int kind, void* out_any) {
  auto* any = static_cast<tvm::ffi::Any*>(out_any);
  handle->impl->GetAttr(dev, static_cast<tvm::runtime::DeviceAttrKind>(kind), any);
}

TVM_DLL size_t TVMDeviceAPIGetDataSize(TVMDeviceAPI* handle, const DLTensor* arr,
                                       const char* mem_scope) {
  tvm::ffi::Optional<tvm::ffi::String> opt_scope = std::nullopt;
  if (mem_scope != nullptr && mem_scope[0] != '\0') {
    opt_scope = tvm::ffi::String(mem_scope);
  }
  return handle->impl->GetDataSize(*arr, opt_scope);
}

TVM_DLL void TVMDeviceAPIGetTargetProperty(TVMDeviceAPI* handle, DLDevice dev, const char* property,
                                           void* out_any) {
  auto* any = static_cast<tvm::ffi::Any*>(out_any);
  handle->impl->GetTargetProperty(dev, std::string(property), any);
}

TVM_DLL void* TVMDeviceAPIAllocDataSpace(TVMDeviceAPI* handle, DLDevice dev, size_t nbytes,
                                         size_t alignment, DLDataType type_hint) {
  return handle->impl->AllocDataSpace(dev, nbytes, alignment, type_hint);
}

TVM_DLL void* TVMDeviceAPIAllocDataSpaceND(TVMDeviceAPI* handle, DLDevice dev, int ndim,
                                           const int64_t* shape, DLDataType dtype,
                                           const char* mem_scope) {
  tvm::ffi::Optional<tvm::ffi::String> opt_scope = std::nullopt;
  if (mem_scope != nullptr && mem_scope[0] != '\0') {
    opt_scope = tvm::ffi::String(mem_scope);
  }
  return handle->impl->AllocDataSpace(dev, ndim, shape, dtype, opt_scope);
}

TVM_DLL void TVMDeviceAPIFreeDataSpace(TVMDeviceAPI* handle, DLDevice dev, void* ptr) {
  handle->impl->FreeDataSpace(dev, ptr);
}

TVM_DLL void TVMDeviceAPICopyDataFromTo(TVMDeviceAPI* handle, DLTensor* from, DLTensor* to,
                                        TVMStreamHandle stream) {
  handle->impl->CopyDataFromTo(from, to, stream);
}

TVM_DLL TVMStreamHandle TVMDeviceAPICreateStream(TVMDeviceAPI* handle, DLDevice dev) {
  return handle->impl->CreateStream(dev);
}

TVM_DLL void TVMDeviceAPIFreeStream(TVMDeviceAPI* handle, DLDevice dev, TVMStreamHandle stream) {
  handle->impl->FreeStream(dev, stream);
}

TVM_DLL void TVMDeviceAPIStreamSync(TVMDeviceAPI* handle, DLDevice dev, TVMStreamHandle stream) {
  handle->impl->StreamSync(dev, stream);
}

TVM_DLL void TVMDeviceAPISetStream(TVMDeviceAPI* handle, DLDevice dev, TVMStreamHandle stream) {
  handle->impl->SetStream(dev, stream);
}

TVM_DLL TVMStreamHandle TVMDeviceAPIGetCurrentStream(TVMDeviceAPI* handle, DLDevice dev) {
  return handle->impl->GetCurrentStream(dev);
}

TVM_DLL void TVMDeviceAPISyncStreamFromTo(TVMDeviceAPI* handle, DLDevice dev,
                                          TVMStreamHandle event_src, TVMStreamHandle event_dst) {
  handle->impl->SyncStreamFromTo(dev, event_src, event_dst);
}

TVM_DLL void* TVMDeviceAPIAllocWorkspace(TVMDeviceAPI* handle, DLDevice dev, size_t nbytes,
                                         DLDataType type_hint) {
  return handle->impl->AllocWorkspace(dev, nbytes, type_hint);
}

TVM_DLL void TVMDeviceAPIFreeWorkspace(TVMDeviceAPI* handle, DLDevice dev, void* ptr) {
  handle->impl->FreeWorkspace(dev, ptr);
}

TVM_DLL int TVMDeviceAPISupportsDevicePointerArithmeticsOnHost(TVMDeviceAPI* handle) {
  return handle->impl->SupportsDevicePointerArithmeticsOnHost() ? 1 : 0;
}
}
