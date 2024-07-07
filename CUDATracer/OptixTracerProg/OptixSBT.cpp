#include "OptixSBT.hpp"

#include "OptixCommon.h"

#include "OptixTraceable.hpp"

static constexpr std::size_t GetHitGroupRecordSize() {
    auto words 
        = (sizeof(CUDATracer::OptixSBT::HitgroupRecord) + (OPTIX_SBT_RECORD_ALIGNMENT - 1)) 
        / OPTIX_SBT_RECORD_ALIGNMENT;

    return words * OPTIX_SBT_RECORD_ALIGNMENT;
}

namespace CUDATracer
{
    
    OptixSBT::OptixSBT(
        const OptixAPI& api,
        OptixProgramGroup _prog_raygen, 
        OptixProgramGroup _prog_raymiss,
        OptixProgramGroup _prog_rayhit
    )
        : _api(api)
        , _sbt()
        , _progHit(_prog_rayhit)
        , _pSbtRayHitBuffer(nullptr)
    {

        RaygenRecord* raygenRecord = (RaygenRecord*)_sbtRayGenBuffer.mutable_cpu_data();
        *raygenRecord = {};
        OPTIX_CHECK(_api.Get().optixSbtRecordPackHeader(_prog_raygen, raygenRecord));

        MissRecord*  missRecord = (MissRecord*)_sbtRayMissBuffer.mutable_cpu_data();
        *missRecord = {};
        OPTIX_CHECK(_api.Get().optixSbtRecordPackHeader(_prog_raymiss, missRecord));

        //HitgroupRecord* hitgroupRecord = (HitgroupRecord*)_sbtRayHitBuffer.mutable_cpu_data();
        //*hitgroupRecord = {};
        //OPTIX_CHECK(_api.Get().optixSbtRecordPackHeader(_prog_rayhit, hitgroupRecord));
    

        _sbt.raygenRecord = (CUdeviceptr)_sbtRayGenBuffer.gpu_data();

        _sbt.missRecordBase = (CUdeviceptr)_sbtRayMissBuffer.gpu_data();
        _sbt.missRecordStrideInBytes = sizeof(MissRecord);
        _sbt.missRecordCount = 1;

        //_sbt.hitgroupRecordBase = (CUdeviceptr)_sbtRayHitBuffer.gpu_data();
        _sbt.hitgroupRecordStrideInBytes = GetHitGroupRecordSize();
        //_sbt.hitgroupRecordCount = 1;
    
    }


    OptixSBT::~OptixSBT() {

    }

    void OptixSBT::Update(OptixTraceable& traceable, const Scene& scene) {

        auto& objects = scene.objects;

        auto recordCnt = objects.size();
        auto bufferSize = recordCnt * GetHitGroupRecordSize();

        if(_pSbtRayHitBuffer == nullptr || _pSbtRayHitBuffer->size() < bufferSize) {
            delete _pSbtRayHitBuffer;
            _pSbtRayHitBuffer = new CUDABuffer(bufferSize);
        }
        auto pRec = (HitgroupRecord*)_pSbtRayHitBuffer->mutable_cpu_data();
        
        auto d_indices  = (std::size_t)traceable.GetIndexBuffer().gpu_data();
        std::uint32_t currInd = 0;

        for(auto& obj : objects) {
            OPTIX_CHECK(_api.Get().optixSbtRecordPackHeader(_progHit, pRec));  
            
            CUdeviceptr indexBuffer = d_indices + currInd * sizeof(uint32_t);

            pRec->data.vertex = (const Vertex*)traceable.GetVertexBuffer().gpu_data();
            pRec->data.index  = (const Math::vec3i*)indexBuffer;
            pRec->data.material = obj.GetMat();

            pRec = (HitgroupRecord*)( ((std::size_t)pRec) + GetHitGroupRecordSize() );

            
            currInd += obj.GetIndices().size();
        }

        _sbt.hitgroupRecordBase = (CUdeviceptr)_pSbtRayHitBuffer->gpu_data();
        _sbt.hitgroupRecordCount = objects.size();
    }

    const OptixShaderBindingTable& OptixSBT::GetBindingTable() const {
        //auto devPtr = (std::size_t)_sbtBuffer.gpu_data();

        //_sbt.raygenRecord = (CUdeviceptr)devPtr + GetRayGenRecordOffset();
        //_sbt.missRecordBase = (CUdeviceptr)devPtr + GetMissRecordOffset();
        //_sbt.hitgroupRecordBase = (CUdeviceptr)devPtr + GetHitGroupRecordOffset();

        return _sbt;
    }

} // namespace CUDATracer

