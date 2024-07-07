#pragma once

#include "OptixAPI.hpp"

#include "CUDABuffer.hpp"
#include "Shaders/ShaderTypes.h"

namespace CUDATracer
{
    class OptixTraceable;
    class OptixSBT {

    public:

        /*! SBT record for a raygen program */
        struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            // just a dummy value - later examples will use more interesting
            // data here
            void* data;
        };

        /*! SBT record for a miss program */
        struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            // just a dummy value - later examples will use more interesting
            // data here
            void* data;
        };

        /*! SBT record for a hitgroup program */
        struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            TriangleMeshSBTData data;
        };

    private:
        const OptixAPI& _api;        
        OptixShaderBindingTable    _sbt;
        TypedBuffer<RaygenRecord>   _sbtRayGenBuffer;
        TypedBuffer<MissRecord>     _sbtRayMissBuffer;
        CUDABuffer* _pSbtRayHitBuffer;

        OptixProgramGroup _progHit;

    public:

        OptixSBT(
            const OptixAPI& api,
            OptixProgramGroup _prog_raygen, 
            OptixProgramGroup _prog_raymiss,
            OptixProgramGroup _prog_rayhit
            );
        ~OptixSBT();

        void Update(OptixTraceable& traceable, const Scene& scene);

        const OptixShaderBindingTable& GetBindingTable() const;
    };

} // namespace CUDATracer

