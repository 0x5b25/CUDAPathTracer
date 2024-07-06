#include "CUDABuffer.hpp"


// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
//inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
//    CUDA_CHECK(cudaMallocHost(ptr, size));
//    *use_cuda = true;
//    return;
//}

//inline void CaffeFreeHost(void* ptr, bool use_cuda) {
//    CUDA_CHECK(cudaFreeHost(ptr));
//    return;
//}


static void gpu_memcpy(const size_t N, const void* X, void* Y) {
    if (X != Y) {
        CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
    }
}

namespace CUDATracer {

    CUDABuffer::CUDABuffer()
        : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {
#ifndef NDEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
#endif
    }

    CUDABuffer::CUDABuffer(CUDABuffer&& another)
        : cpu_ptr_(another.cpu_ptr_)
        , gpu_ptr_(another.gpu_ptr_)
        , size_(another.size_)
        , head_(another.head_)
        , own_cpu_data_(another.own_cpu_data_)
        , own_gpu_data_(another.own_gpu_data_)
#ifndef NDEBUG
        , device_(another.device_)
#endif
    {
        another.cpu_ptr_ = nullptr;
        another.gpu_ptr_ = nullptr;
        another.size_ = 0;
        another.head_ = UNINITIALIZED;
        another.own_cpu_data_ = false;
        another.own_gpu_data_ = false;

#ifndef NDEBUG
        another.device_ = -1;
#endif
    }

    CUDABuffer::CUDABuffer(size_t size)
        : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {

#ifndef NDEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
#endif
    }

    CUDABuffer::~CUDABuffer() {
        release_resources();
    }

    void CUDABuffer::release_resources() {
        check_device();
        if (cpu_ptr_ && own_cpu_data_) {
            CUDA_CHECK(cudaFreeHost(cpu_ptr_));
        }

        if (gpu_ptr_ && own_gpu_data_) {
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }

    }

    CUDABuffer& CUDABuffer::operator=(CUDABuffer&& another) {

        release_resources();

        cpu_ptr_  =  another.cpu_ptr_;
        gpu_ptr_ = another.gpu_ptr_;
        size_ = another.size_;
        head_ = another.head_;
        own_cpu_data_ = another.own_cpu_data_;
        own_gpu_data_ = another.own_gpu_data_;
    
        another.cpu_ptr_ = nullptr;
        another.gpu_ptr_ = nullptr;
        another.size_ = 0;
        another.head_ = UNINITIALIZED;
        another.own_cpu_data_ = false;
        another.own_gpu_data_ = false;

#ifndef NDEBUG
        device_ = another.device_;
        another.device_ = -1;
#endif

        return *this;
    }

    inline void CUDABuffer::to_cpu() {
        check_device();
        switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_));
            memset(cpu_ptr_, 0, size_);
            head_ = HEAD_AT_CPU;
            own_cpu_data_ = true;
            break;
        case HEAD_AT_GPU:
            if (cpu_ptr_ == nullptr) {
                CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_));
                own_cpu_data_ = true;
            }
            CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, 
                cudaMemcpyKind::cudaMemcpyDeviceToHost));
            //gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
            head_ = SYNCED;

            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }

    inline void CUDABuffer::to_gpu() {
        check_device();
        switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
            head_ = HEAD_AT_GPU;
            own_gpu_data_ = true;
            break;
        case HEAD_AT_CPU:
            if (gpu_ptr_ == nullptr) {
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                own_gpu_data_ = true;
            }
            //gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
            CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_,
                cudaMemcpyKind::cudaMemcpyHostToDevice));
            head_ = SYNCED;
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
        }

    }

    const void* CUDABuffer::cpu_data() {
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    void CUDABuffer::set_cpu_data(void* data) {
        check_device();
        CHECK(data);
        if (own_cpu_data_) {
            CUDA_CHECK(cudaFreeHost(cpu_ptr_));
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }

    const void* CUDABuffer::gpu_data() {
        check_device();
        to_gpu();
        return (const void*)gpu_ptr_;

    }

    void CUDABuffer::set_gpu_data(void* data) {
        check_device();
        CHECK(data);
        if (own_gpu_data_) {
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false;
    }

    void* CUDABuffer::mutable_cpu_data() {
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void* CUDABuffer::mutable_gpu_data() {
        check_device();
        to_gpu();
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;
    }

    void CUDABuffer::async_gpu_push(const cudaStream_t& stream) {

        check_device();
        CHECK(head_ == HEAD_AT_CPU);
        if (gpu_ptr_ == nullptr) {
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            own_gpu_data_ = true;
        }
        const cudaMemcpyKind put = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
        // Assume caller will synchronize on the stream before use
        head_ = SYNCED;
    }

    void CUDABuffer::check_device() {
#ifndef NDEBUG
        int device;
        cudaGetDevice(&device);
        CHECK(device == device_);
        if (gpu_ptr_ && own_gpu_data_) {
            cudaPointerAttributes attributes;
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
            CHECK(attributes.device == device_);
            if (attributes.device != device_) {
                __debugbreak();
            }
        }
#endif
    }
}
