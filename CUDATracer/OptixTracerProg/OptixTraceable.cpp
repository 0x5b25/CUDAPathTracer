#include "OptixTraceable.hpp"


#include "OptixCommon.h"

namespace CUDATracer {

    OptixTraceable::OptixTraceable(
        const OptixAPI& api,
        OptixDeviceContext ctx,        
        OptixProgramGroup _prog_raygen, 
        OptixProgramGroup _prog_raymiss,
        OptixProgramGroup _prog_rayhit,
        const Scene& scene
    )
        : _api(api)
        , _ctx(ctx)
        , _sbt(api, _prog_raygen, _prog_raymiss, _prog_rayhit)
    {
        std::size_t vertCnt = 0, indCnt = 0;
        for (auto& obj : scene.objects) {
            vertCnt += obj.GetVertices().size();
            indCnt += obj.GetIndices().size();
        }

        _vertBuffer = new CUDABuffer(vertCnt * sizeof(Vertex));
        _indBuffer = new CUDABuffer(indCnt * sizeof(uint32_t));

        auto pVert = (Vertex*)  _vertBuffer->mutable_cpu_data();
        auto pInd  = (uint32_t*)_indBuffer->mutable_cpu_data();
        std::uint32_t currVert = 0;

        for (auto& obj : scene.objects) {
            auto indBase = currVert;
            for (auto& vert : obj.GetVertices()) {
                pVert[currVert] = vert;
                currVert++;
            }

            for (auto& ind : obj.GetIndices()) {
                *pInd = indBase + ind;
                pInd++;
            }
        }
        
        // create local variables, because we need a *pointer* to the
        // device pointers
        auto d_vertices = (std::size_t)_vertBuffer->gpu_data();
        auto d_indices  = (std::size_t)_indBuffer->gpu_data();

        OptixTraversableHandle asHandle { };

        // ==================================================================
        // triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInputs { };

        CUdeviceptr vertBuffers[1]{ d_vertices };
  
        std::uint32_t currInd = 0;
        for (unsigned i = 0; i < scene.objects.size(); i++) {
            auto& obj = scene.objects[i];

            CUdeviceptr indexBuffer = d_indices + currInd * sizeof(uint32_t);

            OptixBuildInput triangleInput{};
        
            triangleInput.type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
            triangleInput.triangleArray.numVertices = vertCnt;
            triangleInput.triangleArray.vertexBuffers = vertBuffers;

            triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
            triangleInput.triangleArray.numIndexTriplets = (int)obj.GetIndices().size()/3;
            triangleInput.triangleArray.indexBuffer = indexBuffer;

            uint32_t triangleInputFlags[1] = { 0 };

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInput.triangleArray.flags = triangleInputFlags;
            triangleInput.triangleArray.numSbtRecords = 1;
            triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
            triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            triangleInputs.push_back(triangleInput);

            currInd += obj.GetIndices().size();
        }
        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
            | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS //For getting vertex data
            ;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(_api.Get().optixAccelComputeMemoryUsage(
            _ctx,
            &accelOptions,
            triangleInputs.data(),
            triangleInputs.size(),  // num_build_inputs
            &blasBufferSizes
        ));

        // ==================================================================
        // prepare compaction
        // ==================================================================

        CUDABuffer compactedSizeBuffer(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = (CUdeviceptr)compactedSizeBuffer.gpu_data();

        // ==================================================================
        // execute build (main stage)
        // ==================================================================

        CUDABuffer tempBuffer(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(_api.Get().optixAccelBuild(
            _ctx,
            /* stream */0,
            &accelOptions,
            triangleInputs.data(),
            triangleInputs.size(),
            (CUdeviceptr)tempBuffer.gpu_data(),
            blasBufferSizes.tempSizeInBytes,

            (CUdeviceptr)outputBuffer.gpu_data(),
            blasBufferSizes.outputSizeInBytes,

            &asHandle,

            &emitDesc, 1
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize = *(uint64_t*)compactedSizeBuffer.cpu_data();

        _asBuffer = new CUDABuffer(compactedSize);
        OPTIX_CHECK(_api.Get().optixAccelCompact(
            _ctx,
            /*stream:*/0,
            asHandle,
            (CUdeviceptr)_asBuffer->gpu_data(),
            compactedSize,
            &_asHandle));
        CUDA_CHECK(cudaDeviceSynchronize());

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        //outputBuffer.free(); // << the UNcompacted, temporary output buffer
        //tempBuffer.free();
        //compactedSizeBuffer.free();

        _sbt.Update(*this, scene);
    }


    OptixTraceable::~OptixTraceable() {
        delete _asBuffer;
        delete _vertBuffer;
        delete _indBuffer;
    }


}
