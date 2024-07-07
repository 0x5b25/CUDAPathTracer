#include "OptixTracerProg.hpp"

#include <cstring>

#include "OptixCommon.h"
#include "OptixTraceable.hpp"

#include "Shaders/DefaultShader.compiled.h"

namespace CUDATracer
{

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
    
    
    bool OptixTracerProg::InitOptiX()
    {
        //CUresult cuRes = cuInit(0); // Initialize CUDA driver API.
        //if (cuRes != CUDA_SUCCESS)
        //{
        //  std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
        //  return false;
        //}

        //getSystemInformation(); // Get device attributes of all found devices. Fills m_deviceAttributes.

        CUdevice device = 0;

        CUresult cuRes = cuDeviceGet(&device, 0); // Get the CUdevice from device ordinal 0. (These are usually identical.)
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuDeviceGet() failed: " << cuRes << '\n';
          return false;
        }

        cuRes = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device); // DEBUG What is the best CU_CTX_SCHED_* setting here.
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
          return false;
        }

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        cuRes = cuStreamCreate(&m_cudaStream, CU_STREAM_DEFAULT);
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
          return false;
        }


        OptixDeviceContextOptions options = {};

        options.logCallbackFunction = &Logger::callback;
        options.logCallbackData     = &m_logger;
        options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.

        auto res = _api.Get().optixDeviceContextCreate(m_cudaContext, &options, &m_context);
        if (res != OPTIX_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
          return false;
        }

        //return InitRenderer(); // Initialize all the rest.

        return true;
    }

    
    void OptixTracerProg::TearDownOptiX() {

        
        OPTIX_CHECK( _api.Get().optixDeviceContextDestroy(m_context) );
        
        CU_CHECK( cuStreamDestroy(m_cudaStream) );
        CU_CHECK( cuCtxDestroy(m_cudaContext) );
    }

    
    bool OptixTracerProg::InitRenderer() {

        std::cout << "#osc: setting up module ..." << std::endl;
        /**************************************
         *        createModule                *
         **************************************/

        OptixModuleCompileOptions moduleCompileOptions{ };

        moduleCompileOptions.maxRegisterCount = 50;

#ifdef NDEBUG
        //Release config
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else // !NDEBUG
        //Debug config
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif // !NDEBUG

        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
        //const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof(log);
//#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(_api.Get().optixModuleCreate(m_context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            (const char*)DefaultShader,//ptxCode.c_str(),
            sizeof(DefaultShader),
            log, &sizeof_log,
            &_module
        ));
//#else
//        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
//            &moduleCompileOptions,
//            &pipelineCompileOptions,
//            ptxCode.c_str(),
//            ptxCode.size(),
//            log,      // Log string
//            &sizeof_log,// Log string sizse
//            &module
//        ));
//#endif
        if (sizeof_log > 1) PRINT(log);

        auto _CreateProgram = [&](OptixProgramGroupKind kind, const std::string& entry1, const std::string& entry2)-> OptixProgramGroup {

            OptixProgramGroupOptions pgOptions { };
            OptixProgramGroupDesc pgDesc { };
            pgDesc.kind = kind;// OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            switch (kind) {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                pgDesc.raygen.module = _module;
                pgDesc.raygen.entryFunctionName = entry1.c_str();// "__raygen__renderFrame";
                break;

            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                pgDesc.miss.module = _module;
                pgDesc.miss.entryFunctionName = entry1.c_str();// "__raygen__renderFrame";
                break;

            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                pgDesc.hitgroup.moduleCH = _module;
                pgDesc.hitgroup.entryFunctionNameCH = entry1.c_str();
                pgDesc.hitgroup.moduleAH = _module;
                pgDesc.hitgroup.entryFunctionNameAH = entry2.c_str();
                break;
            }

            OptixProgramGroup pg;

            char log[2048];
            size_t sizeof_log = sizeof(log);
            auto res = _api.Get().optixProgramGroupCreate(m_context,
                &pgDesc,
                1,
                &pgOptions,
                log, &sizeof_log,
                &pg
            );
            if (res == OPTIX_SUCCESS) {
                return pg;
            }
            else {
                if (sizeof_log > 1) PRINT(log);
                DebugBreak();
                return nullptr;
            }
        };

        std::cout << "#osc: creating raygen programs ..." << std::endl;
        _prog_raygen = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "__raygen__renderFrame", "");
        std::cout << "#osc: creating miss programs ..." << std::endl;
        _prog_raymiss = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_MISS, "__miss__radiance", "");
        std::cout << "#osc: creating hitgroup programs ..." << std::endl;
        _prog_rayhit = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "__closesthit__radiance", "__anyhit__radiance");

        std::cout << "#osc: setting up optix pipeline ..." << std::endl;

        /**************************************
         *        createPipeline              *
         **************************************/

        std::vector<OptixProgramGroup> programGroups{
            _prog_raygen, _prog_raymiss, _prog_rayhit
        };

        OptixPipelineLinkOptions pipelineLinkOptions { };

        pipelineLinkOptions.maxTraceDepth = RT_TRACE_DEPTH;

        memset(log, 0, sizeof(log));
        sizeof_log = sizeof(log);
        OPTIX_CHECK(_api.Get().optixPipelineCreate(m_context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &_pipeline
        ));
        if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(_api.Get().optixPipelineSetStackSize
        (/* [in] The pipeline to configure the stack size for */
            _pipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            2 * 1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1));

        std::cout << "#osc: building SBT ..." << std::endl;

        /**************************************
         *        buildSBT                    *
         **************************************/
        //_pSbt = new OptixSBT(_api, _prog_raygen, _prog_raymiss, _prog_rayhit);

        //_launchParamsBuffer = std::move(CUDABuffer(sizeof(LaunchParams)));
        _pLaunchParamsBuffer = new TypedBuffer<LaunchParams>();
        _pLaunchParamsBuffer->GetMutable<0>() = {};
        std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

        //std::cout << GDT_TERMINAL_GREEN;
        std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
        //std::cout << GDT_TERMINAL_DEFAULT;

        return true;
    }


    void OptixTracerProg::TearDownRenderer() {
        OPTIX_CHECK(_api.Get().optixPipelineDestroy(_pipeline));

        for (auto& pg : { _prog_raygen, _prog_raymiss, _prog_rayhit }) {
            OPTIX_CHECK(_api.Get().optixProgramGroupDestroy(pg));
        }

        OPTIX_CHECK(_api.Get().optixModuleDestroy(_module));

        //delete _pSbtRecords;
        delete _pLaunchParamsBuffer;        
    }


    OptixTracerProg::OptixTracerProg() 
        : IPathTracer()
        , m_logger(std::cerr)
    {
        //cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192);
        InitOptiX();
        InitRenderer();
    }


    OptixTracerProg:: ~OptixTracerProg() {
        TearDownRenderer();

        TearDownOptiX();
    }

    
    ITraceable* OptixTracerProg::CreateTraceable(const Scene& scene) const {
        return new OptixTraceable(
            _api, m_context, _prog_raygen, _prog_raymiss, _prog_rayhit, scene);
    }

    void OptixTracerProg::Trace(
        ITraceable& scn, TypedBuffer<PathTraceSettings>& settings,
        float* accBuffer, char* buffer
    ){
        auto stdata = (PathTraceSettings*)settings.cpu_data();
        auto& optixScn = static_cast<OptixTraceable&>(scn);

        auto w = stdata->viewportWidth;
        auto h = stdata->viewportHeight;

        if (w * h == 0) return;
//
//
        //const unsigned batchW = BATCH_W, batchH = BATCH_H;
        //const unsigned blockW = (w + batchW - 1) / batchW;
        //const unsigned blockH = (h + batchH - 1) / batchH;
//
        //uint3 blkPerGrid{ blockW, blockH, 1 };
        ////uint3 blkPerGrid{ 2, 2, 1 };
        //uint3 threadPerBlk{batchW, batchH, 1};
//
        //auto scene_gpu = (SceneInfoHeader*)scn.GetGPUData();
        //auto settings_gpu = (PathTraceSettings*)settings.gpu_data();
        ////auto cache_gpu = (std::uint8_t*)_rayCache->gpu_data();
        //RTProgram prog{ scene_gpu, settings_gpu};
//
        //GenRay << <blkPerGrid, threadPerBlk >> > (prog, accBuffer, buffer);

        // sanity check: make sure we launch only after first resize is
    // already done:
        {
            auto& launchParams = _pLaunchParamsBuffer->GetMutable<0>();
            launchParams.frame.size.x = w;
            launchParams.frame.size.y = h;
            launchParams.frame.colorBuffer = buffer;
            launchParams.frame.accBuffer = accBuffer;
            launchParams.frame.frameID = stdata->frameID;
            launchParams.camera = stdata->cam;
            launchParams.traversable = optixScn.GetHandle();
        }


        //Upload to GPU
        auto devPtr = (CUdeviceptr)_pLaunchParamsBuffer->gpu_data();

        //{
        //    auto& launchParams = _pLaunchParamsBuffer->GetMutable<0>();
        //    launchParams.frameID++;
        //}
        OPTIX_CHECK(_api.Get().optixLaunch(/*! pipeline we're launching launch: */
            _pipeline, m_cudaStream,
            /*! parameters and SBT */
            devPtr,
            _pLaunchParamsBuffer->size(),
            //&_sbt,
            &optixScn.GetSBT(),
            /*! dimensions of the launch: */
            w,
            h,
            1
        ));
        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_CHECK(cudaDeviceSynchronize());
        
    }

} // namespace CUDAPathTracer

