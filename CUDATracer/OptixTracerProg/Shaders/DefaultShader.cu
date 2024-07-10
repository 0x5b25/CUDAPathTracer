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

    __both__ Math::vec3f RandDirInsideCone(
        LCG<>& randGen,
        const Math::vec3f& dir,
        float size
    ) {
        //Calculate "LookAt" transform matrix
        Math::vec2f right_(dir.z, -dir.x);
        float rlen_ = Math::length(right_);
        Math::vec3f right = rlen_ <= 0 ?
            (Math::vec3f{ 0,0,1 }) :
            (Math::vec3f{ right_.x, 0, right_.y } / rlen_);
        Math::vec3f up = Math::cross(right, dir);

        //assert(dot(normal, dir) > 0); //Same direction

        //Pick point on unit sphere with pole = x axis
        float u = randGen.NextFloat();
        float v = randGen.NextFloat();

        float theta = M_PI * 2 * u;
        //float phi = std::acos(1 - 2*v*size);

        float sx = 1 - 2 * v * size;
        float sy = std::sqrt(1 - sx * sx) * std::cos(theta);
        float sz = std::sqrt(1 - sx * sx) * std::sin(theta);

        //Rotate pole
        Math::vec3f sdir =
            sx * dir +
            sy * up +
            sz * right
            ;

        auto ld = Math::length(dir);
        auto lu = Math::length(up);
        auto lr = Math::length(right);

        auto ls = Math::length(sdir);
        auto lsr = sx * sx + sy * sy + sz * sz;

        //assert(glm::length(sdir) <= 1 + 1e-3);
        return sdir;
    }

    __both__ Math::vec3f GGXRandDirInsideCone(
        LCG<>& randGen,
        const Math::vec3f& dir,
        float size
    ) {
        //Calculate "LookAt" transform matrix
        Math::vec2f right_(dir.z, -dir.x);
        float rlen_ = Math::length(right_);
        Math::vec3f right = rlen_ <= 0 ?
            (Math::vec3f{ 0,0,1 }) :
            (Math::vec3f{ right_.x, 0, right_.y } / rlen_);
        Math::vec3f up = Math::cross(right, dir);

        //assert(dot(normal, dir) > 0); //Same direction

        //Pick point on unit sphere with pole = x axis
        float u = randGen.NextFloat();
        float v = randGen.NextFloat();

        float theta = M_PI * 2 * u;
        float phi = atan2f(size * sqrtf(v),sqrtf(1-v));


        float sx = 1 - sin(phi);
        float sy = std::sqrt(1 - sx * sx) * std::cos(theta);
        float sz = std::sqrt(1 - sx * sx) * std::sin(theta);

        //Rotate pole
        Math::vec3f sdir =
            sx * dir +
            sy * up +
            sz * right
            ;

        auto ld = Math::length(dir);
        auto lu = Math::length(up);
        auto lr = Math::length(right);

        auto ls = Math::length(sdir);
        auto lsr = sx * sx + sy * sy + sz * sz;

        //assert(glm::length(sdir) <= 1 + 1e-3);
        return sdir;
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
        auto &prd = *getPRD<PRD>();
        int ttl = prd.ttl;

        if(prd.ttl < 1){
             // set to constant white as background color
            prd.rc = Math::vec3f(.001f);
        }
        else{
            //Calculate normals

            const TriangleMeshSBTData &sbtData
                = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

            auto mat = sbtData.material;
            
            const int   primID = optixGetPrimitiveIndex();
            const Math::vec3i index  = sbtData.index[primID];
            auto vert0 = sbtData.vertex[index.x];
            auto vert1 = sbtData.vertex[index.y];
            auto vert2 = sbtData.vertex[index.z];

            const Math::vec3f v0 = vert0.position;
            const Math::vec3f v1 = vert1.position;
            const Math::vec3f v2 = vert2.position;
            
            const Math::vec3f rayDir = optixGetWorldRayDirection();
            Math::vec3f hitPosition = rayDir * optixGetRayTmax() + Math::vec3f(optixGetWorldRayOrigin());
            

            auto v0v1 = v1 - v0;
            auto v0v2 = v2 - v0;
            auto pvec = Math::cross(rayDir, v0v2);
            float det = Math::dot(v0v1, pvec);

            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            //if (det < FLT_EPSILON) continue;
            // ray and triangle are parallel if det is close to 0
            //if (fabs(det) < FLT_EPSILON) continue;
            float invDet = 1 / det;

            auto tvec = Math::vec3f(optixGetWorldRayOrigin()) - v0;
            auto u = Math::dot(tvec, pvec) * invDet;
            //if (u < 0 || u > 1) continue;

            auto qvec = Math::cross(tvec, v0v1);
            auto v = Math::dot(rayDir, qvec) * invDet;
            //if (v < 0 || u + v > 1) continue;

            Math::vec3f weights = { 1 - u - v, u,v };

            const Math::vec2f uv = vert0.uv * weights.x +
                                   vert1.uv * weights.y +
                                   vert2.uv * weights.z;
            const Math::vec3f materalNorm =
                Math::normalize(vert0.normal) * weights.x +
                Math::normalize(vert1.normal) * weights.y +
                Math::normalize(vert2.normal) * weights.z;

            bool hitFrontFace = dot(materalNorm, rayDir) < 0.0f;

            float offset = 1e-4f;

            auto surfaceNorm = hitFrontFace? materalNorm : -materalNorm;

            ////Back face handling
            //// If the dot product is positive the ray points into the same hemisphere as the face normal, means it's hitting a backface.
            //// Also remove edge-on cases with the equal comparison.
            //if (dot(norm, rayDir) > 0.0f)
            //{
            //    //return; // Do not calculate anything more for this triangle in the intersection program.
            //    offset = -1e-4f;
            //}

            //Calculate reflect dir
            auto norm = GGXRandDirInsideCone(*(prd.randGen), surfaceNorm, mat.roughness);
            auto dir = Math::reflect(rayDir, norm);

            auto rayOrigin = hitPosition + surfaceNorm * offset;
            //Launch reflect ray
            prd.ttl = ttl-1;
            
            uint32_t u0 = optixGetPayload_0(), 
                     u1 = optixGetPayload_1();

            optixTrace(
                optixLaunchParams.traversable,
                rayOrigin,
                dir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                SURFACE_RAY_TYPE,             // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                SURFACE_RAY_TYPE,             // missSBTIndex 
                u0, u1
            );

            //Get color from ray
            auto c = Math::clamp( prd.rc,Math::vec3f(0),Math::vec3f(Math::infty()));;

            //Calculate refraction and metalicity
            //Refraction index(incident)
            //float n1 = prd->ni;
            float n1 = 1.f;
            //Refraction index(refraction)
            float n2 = mat.n;
            n2 += FLT_EPSILON;
            //Outbound ray, fetch prev n

            if (!hitFrontFace) {
                auto t = n1; n1 = n2; n2 = t;
            }
            //Actual normal
            //auto sampleNorm = Math::normalize(raySelDir - ray.dir);

            //Schlick's approximation of Fresnel equation
            //r(x) = r0 + (1 - r0)(1 - cos(x))^5
            //r0 = ((n1 - n2)/(n1 + n2))^2
            float r0 = (n1 - n2) / (n1 + n2); r0 *= r0;

            //float rr = r0 + (1 - r0) * (1 - Math::dot(-ray.dir, sampleNorm))^5;
            float rr = (1 - Math::dot(-rayDir, norm));
            rr = powf(rr, 5);
            rr = r0 + (1 - r0) * rr;

            rr = __saturatef(rr);

            //Opt-in metallic param
            rr = mat.metallic * 1.f + (1 - mat.metallic) * rr;

            float rt = 1 - rr;

            auto reflection_absorb = mat.metallic * mat.color + (1 - mat.metallic) * Math::vec3f(1.f);
            auto currColor = reflection_absorb * c * rr;

            Math::vec3f refractRayDir, refractRayOrigin;

            if(mat.opacity < 1.f){
                auto front = (rayDir + dir) / 2;
                auto down = rayDir - front;
                float down_len = Math::length(down);
                down = down_len < 1e-6f?Math::vec3f(1,0,0):(down/down_len);
                //Snell's law
                // ni sin(xi) = nt sin(xt)
                //               ni sin(xi)
                //  sin(xt) = ----------------
                //                  nt

                //float front_refr_len = n1 * Math::length(front) / n2;
                auto front_refr = front * (n1 / n2);
                float front_refr_len = Math::length(front_refr);
                if(front_refr_len > 1) front_refr_len = 1;

                float down_refr_len = sqrt(1 - front_refr_len * front_refr_len);

                auto refr_dir = Math::normalize(front_refr + down * (down_refr_len));

                refractRayDir = refr_dir;
                refractRayOrigin = hitPosition - surfaceNorm * offset;

            }else{
                refractRayDir = RandDirInsideCone(
                    *(prd.randGen), surfaceNorm, 0.49);
                refractRayOrigin = hitPosition + surfaceNorm * offset;

            }

            
            prd.ttl = ttl-1;
            optixTrace(
                optixLaunchParams.traversable,
                refractRayOrigin,
                refractRayDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                SURFACE_RAY_TYPE,             // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                SURFACE_RAY_TYPE,             // missSBTIndex 
                u0, u1
            );


            c = prd.rc;
            //Calculate color
            currColor += mat.color * c * rt * ((mat.opacity == 1.f)? 1.f : ( 1 - mat.opacity));

            currColor += mat.emissiveColor * mat.emissiveStrength;

            ////Collect ray color
            //auto currColor = prd.rc * sbtData.material.color;
            ////auto currColor = prd.rc * Math::vec3f(uv.x, uv.y, 1);
            //currColor += sbtData.material.emissiveColor * sbtData.material.emissiveStrength;
            
            prd.rc = currColor;
            //prd.rc = randomColor(primID);
        }
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
        prd.rc = Math::vec3f(.001f);
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
        
        float weight = 1.0f / (optixLaunchParams.frame.frameID + 1.0f);

        

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
        float pixelSize = 1.0f / edgeLen * fovScale * 2;

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