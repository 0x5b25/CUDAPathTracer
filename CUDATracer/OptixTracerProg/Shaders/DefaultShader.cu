#include <optix_device.h>

#include "ShaderTypes.h"

namespace CUDATracer{

    // for this simple example, we have a single ray type
    enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

    /*! simple 24-bit linear congruence generator */
    template<unsigned int N = 16>
    struct LCG {

        __both__ inline LCG()
        { /* intentionally empty so we can use it in device vars that
             don't allow dynamic initialization (ie, PRD) */
        }
        __both__ inline LCG(unsigned int val0, unsigned int val1)
        {
            init(val0, val1);
        }

        __both__ inline void init(unsigned int val0, unsigned int val1)
        {
            unsigned int v0 = val0;
            unsigned int v1 = val1;
            unsigned int s0 = 0;

            for (unsigned int n = 0; n < N; n++) {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }
            state = v0;
        }

        // Generate random unsigned int in [0, 2^24)
        __both__ inline std::uint32_t NextInt() {
            const uint32_t LCG_A = 1664525u;
            const uint32_t LCG_C = 1013904223u;
            state = (LCG_A * state + LCG_C);
            return state;
        }
        //Generate uniform float in (0,1)
        __both__ inline float NextFloat() {
            //std::uint32_t n;
            //do
            //{
            //    n = NextInt();
            //} while (n <= 0x000001FFU); // If true, then the highest 23 bits must all be zero.
            //std::uint32_t prem = 0x3F800000U | (n >> 9);
            //float res = *((float*)&prem) - 1.0f;
            //assert(res > 0 && res < 1);
            //return res;
            return (NextInt() & 0x00FFFFFF) / (float)0x01000000;
        }

        uint32_t state;
    };

    struct PRD {
        LCG<>* randGen;
        Math::vec3f rc;
        unsigned ttl;/*Time to live, that is how many light bounces left*/
    };

  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;


    static __forceinline__ __device__
    void *unpackPointer( uint32_t i0, uint32_t i1 )
    {
      const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
      void*           ptr = reinterpret_cast<void*>( uptr ); 
      return ptr;
    }

    static __forceinline__ __device__
    void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
    {
      const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
      i0 = uptr >> 32;
      i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD()
    { 
      const uint32_t u0 = optixGetPayload_0();
      const uint32_t u1 = optixGetPayload_1();
      return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
    }

      /*! helper function that creates a semi-random color from an ID */
    static __forceinline__ __device__
    Math::vec3f randomColor(int i)
    {
      int r = unsigned(i)*13*17 + 0x234235;
      int g = unsigned(i)*7*3*5 + 0x773477;
      int b = unsigned(i)*11*19 + 0x223766;
      return Math::vec3f((r&255)/255.f,
                   (g&255)/255.f,
                   (b&255)/255.f);
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------
    
    extern "C" __global__ void __closesthit__radiance() {
        const int   primID = optixGetPrimitiveIndex();
        auto &prd = *getPRD<PRD>();
        prd.rc = randomColor(primID);
    }
    
    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */ }


  
    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------
    
    extern "C" __global__ void __miss__radiance() {
        auto& prd = *getPRD<PRD>();
        // set to constant white as background color
        prd.rc = Math::vec3f(1.f);
    }



    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        //if (optixLaunchParams.frameID == 0 &&
        //    optixGetLaunchIndex().x == 0 &&
        //    optixGetLaunchIndex().y == 0) {
        //  // we could of course also have used optixGetLaunchDims to query
        //  // the launch size, but accessing the optixLaunchParams here
        //  // makes sure they're not getting optimized away (because
        //  // otherwise they'd not get used)
        //  printf("############################################\n");
        //  printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
        //         optixLaunchParams.frame.size.x,
        //         optixLaunchParams.frame.size.y);
        //  printf("############################################\n");
        //}

        // ------------------------------------------------------------------
        // for this example, produce a simple test pattern:
        // ------------------------------------------------------------------

        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto &cam = optixLaunchParams.camera;
        
        float weight = 1.0 / (optixLaunchParams.frame.frameID + 1.0);

        

        //// normalized screen plane position, in [0,1]^2
        //const Math::vec2f screen(Math::vec2f(ix+.5f,iy+.5f)
        //                   / Math::vec2f(optixLaunchParams.frame.size));
        //
        //// generate ray direction
        //Math::vec3f rayDir = Math::normalize(camera.front
        //                         + (screen.x - 0.5f) * camera.right
        //                         + (screen.y - 0.5f) * camera.up);


        float num_x = optixLaunchParams.frame.size.x;
        float num_y = optixLaunchParams.frame.size.y;

        unsigned local_seed = num_x * iy + ix;
        //local_seed += settings->seed;
        
        LCG<> randGen{};
        randGen.init(local_seed, optixLaunchParams.frame.frameID);
        
        // our per-ray data for this example. what we initialize it to
        // won't matter, since this value will be overwritten by either
        // the miss or hit program, anyway
        //Math::vec3f pixelColorPRD(0.f);
        PRD prd{};
        prd.ttl = RT_TRACE_DEPTH;
        prd.randGen = &randGen;

        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer( &prd, u0, u1 );

        float edgeLen = min(num_x, num_y);

        float fovScale = std::atan(cam.fov / 2);
        float pixelSize = 1.0 / edgeLen * fovScale * 2;

        float w = num_x * pixelSize;
        float h = num_y * pixelSize;


        float origin_x = -w / 2;
        float origin_y = -h / 2;

        float offx = randGen.NextFloat() * pixelSize;
        float offy = randGen.NextFloat() * pixelSize;

        float u = (ix * pixelSize + origin_x + offx); //convert to (-1,1)
        float v = (iy * pixelSize + origin_y + offy);

        auto rayDir = cam.front + cam.right * u + cam.up * v;

        rayDir = Math::normalize(rayDir);

        optixTrace(optixLaunchParams.traversable,
                   cam.pos,
                   rayDir,
                   0.f,    // tmin
                   1e20f,  // tmax
                   0.0f,   // rayTime
                   OptixVisibilityMask( 255 ),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                   SURFACE_RAY_TYPE,             // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
                   SURFACE_RAY_TYPE,             // missSBTIndex 
                   u0, u1 );

        auto c = prd.rc;

        //Use accumulate buffer to average over multiple frames:
        //Assume buffer format RGBA and row priori storage
        auto cbuf = (float*)optixLaunchParams.frame.accBuffer;
        const uint32_t addr = (ix+iy*optixLaunchParams.frame.size.x) * 4;
        
        float r = cbuf[addr]     * (1 - weight) + c.x * weight;
        float g = cbuf[addr + 1] * (1 - weight) + c.y * weight;
        float b = cbuf[addr + 2] * (1 - weight) + c.z * weight;

        cbuf[addr] = r;
        cbuf[addr + 1] = g;
        cbuf[addr + 2] = b;
        cbuf[addr + 3] = 1;

        // and write to frame buffer ...
        c = {
            __saturatef(r),
            __saturatef(g),
            __saturatef(b)
        };
        c = c / (c + 1.0f);
        c = Math::pow(c, Math::vec3f(1.0f / 2.2f));

        optixLaunchParams.frame.colorBuffer[addr]     = c.x * 255;
        optixLaunchParams.frame.colorBuffer[addr + 1] = c.y * 255;
        optixLaunchParams.frame.colorBuffer[addr + 2] = c.z * 255;
        optixLaunchParams.frame.colorBuffer[addr + 3] = 255;
    }
}