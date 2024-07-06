#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "CUDACommon.h"

namespace CUDATracer {

    class CUDABuffer {
    public:
        CUDABuffer();
        explicit CUDABuffer(size_t size);
        CUDABuffer(CUDABuffer&& another);

        ~CUDABuffer();
        const void* cpu_data();
        void set_cpu_data(void* data);
        const void* gpu_data();
        void set_gpu_data(void* data);
        void* mutable_cpu_data();
        void* mutable_gpu_data();
        enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
        SyncedHead head() const { return head_; }
        size_t size() const { return size_; }

        void async_gpu_push(const cudaStream_t& stream);

        CUDABuffer& operator=(CUDABuffer&& another);

    private:
        void check_device();
        void release_resources();
        void to_cpu();
        void to_gpu();
        void* cpu_ptr_;
        void* gpu_ptr_;
        size_t size_;
        SyncedHead head_;
        bool own_cpu_data_;
        bool own_gpu_data_;
        int device_;

        DISABLE_COPY_AND_ASSIGN(CUDABuffer);
    };


    template<typename Ty, typename...OtherTy>
    struct VASizeCalc {
        static constexpr std::size_t size = sizeof(Ty) +
            VASizeCalc<OtherTy...>::size;
    };

    template<typename Ty>
    struct VASizeCalc<Ty> {
        static constexpr std::size_t size = sizeof(Ty);
    };

    template<unsigned req_i, unsigned i, unsigned off, typename ...Ty>
    struct VATyGetImpl {};

    template<unsigned i, unsigned off, typename ThisTy, typename ...Ty>
    struct VATyGetImpl<i, i, off, ThisTy, Ty...>
    {
        using type = ThisTy;
        static constexpr unsigned offset = off;
    };


    template<unsigned req_i, unsigned i, unsigned off, typename ThisTy, typename ...Ty>
    struct VATyGetImpl<req_i, i, off, ThisTy, Ty...>
    {
        using NextImpl = VATyGetImpl<req_i, i + 1, off + sizeof(ThisTy), Ty...>;
        using type = typename NextImpl::type;
        static constexpr unsigned offset = NextImpl::offset;
    };


    template<unsigned index, typename ...Ty> using VATyGet = VATyGetImpl<index, 0, 0, Ty...>;

    template<typename Ty, typename ...OtherTy>
    class TypedBuffer : public CUDABuffer {

    public:
        TypedBuffer() : CUDABuffer(VASizeCalc<Ty, OtherTy...>::size){}

        template<unsigned idx>
        const typename VATyGet<idx, Ty, OtherTy...>::type& Get(){
            using TyGet = VATyGet<idx, Ty, OtherTy...>;

            auto ptr = (std::uint8_t*)cpu_data();
            const auto offset = TyGet::offset;
            auto pobj = ptr + offset;
            return *(typename TyGet::type*)pobj;
        }

        template<unsigned idx>
        typename VATyGet<idx, Ty, OtherTy...>::type& GetMutable() {
            using TyGet = VATyGet<idx, Ty, OtherTy...>;

            auto ptr = (std::uint8_t*)mutable_cpu_data();
            const auto offset = TyGet::offset;
            auto pobj = ptr + offset;
            return *(typename TyGet::type*)pobj;
        }

        template<unsigned idx>
        void Set(const typename VATyGet<idx, Ty, OtherTy...>::type& val) {
            using TyGet = VATyGet<idx, Ty, OtherTy...>;

            auto ptr = (std::uint8_t*)mutable_cpu_data();
            const auto offset = TyGet::offset;
            auto pobj = ptr + offset;
            *(typename TyGet::type*)pobj = val;
        }

    };

}