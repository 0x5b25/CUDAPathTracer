#include "CUDATracer.hpp"

#include <string>
#include <float.h>

#include <cuda.h>
//#include <math_functions.h>
#include <cuda_runtime.h>

#include "Types.hpp"
#include "Math/Constants.h"
#include "Math/MathCommon.h"
#include "Math/Vec.hpp"

#ifndef NDEBUG
#define BATCH_W 16
#define BATCH_H 16
#else
#define BATCH_W 16
#define BATCH_H 16
#endif
//#define VISUALIZE_BVH 1

namespace CUDATracer {


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
    __device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
    __device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
    __device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
    __device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
    __device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
    __device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
    __device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
    __device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

    __device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
    __device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }


    template<typename T>
    __device__ __inline__ void swap2(T& a, T& b) { T temp = a; a = b; b = temp; }

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


   
    //MOLLER_TRUMBORE 
    __device__ bool NearestIntersectSubset(
        const MeshInfoHeader& mesh,
        const Ray& ray,
        std::uint32_t* indicesArr,
        unsigned triangleCnt,

        Intersection& outInt
    ) {

        //    return NearestIntersect(ray,
        //        mesh.GetVertices(),
        //        mesh.GetIndices(),
        //        mesh.indicesCnt,
        //        mesh.transform,
        //        outInt
        //    );
        //}
        const Vertex* __restrict__ vertices = mesh.GetVertices();
        const std::uint32_t* __restrict__ indices = mesh.GetIndices();
        //const Math::mat4x4f& transform = mesh.transform;        

        bool hit = false;
        //auto triangleCnt = indicesCnt / 3;

        float tmin = Math::infty();
        Math::vec3f weights;
        int ti;//closest triangle index
        bool frontFacing;
        Math::vec3f pos;
        for (std::uint32_t i = 0; i < triangleCnt; i++) {
            std::uint32_t addr = i * 3;
            const auto& _v0 = vertices[indicesArr[addr]];
            const auto& _v1 = vertices[indicesArr[addr + 1]];
            const auto& _v2 = vertices[indicesArr[addr + 2]];
            //auto v0 = Math::MatMul(transform, _v0.position, 1.f);
            //auto v1 = Math::MatMul(transform, _v1.position, 1.f);
            //auto v2 = Math::MatMul(transform, _v2.position, 1.f);
            const auto& v0 = _v0.position;
            const auto& v1 = _v1.position;
            const auto& v2 = _v2.position;


            // Calculate the vertices relative to the ray origin
            //Math::vec3f& v0 = p0;
            //Math::vec3f& v1 = p1;
            //Math::vec3f& v2 = p2;

            auto v0v1 = v1 - v0;
            auto v0v2 = v2 - v0;
            auto pvec = Math::cross(ray.dir, v0v2);
            float det = Math::dot(v0v1, pvec);

            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            //if (det < FLT_EPSILON) continue;
            // ray and triangle are parallel if det is close to 0
            if (fabs(det) < FLT_EPSILON) continue;
            float invDet = 1 / det;

            auto tvec = ray.origin - v0;
            auto u = Math::dot(tvec, pvec) * invDet;
            if (u < 0 || u > 1) continue;

            auto qvec = Math::cross(tvec, v0v1);
            auto v = Math::dot(ray.dir, qvec) * invDet;
            if (v < 0 || u + v > 1) continue;

            auto t = Math::dot(v0v2, qvec) * invDet;

            if(t > 0){
                hit = true;

                if (t < tmin) {
                    tmin = t;
                    ti = i;
                    weights = { 1 - u - v, u,v };
                    frontFacing = det > 0;
                }
            }

            
        }

        if (hit) {
            //Populate intersection info
            outInt.frontFacing = frontFacing;
            outInt.t = tmin;

            auto addr = ti * 3;
            const auto& v0 = vertices[indicesArr[addr]];
            const auto& v1 = vertices[indicesArr[addr + 1]];
            const auto& v2 = vertices[indicesArr[addr + 2]];

            //auto n0 = Math::MatMul(transform, v0.normal, 0.f);
            //auto n1 = Math::MatMul(transform, v1.normal, 0.f);
            //auto n2 = Math::MatMul(transform, v2.normal, 0.f);

            const auto& n0 = v0.normal;
            const auto& n1 = v1.normal;
            const auto& n2 = v2.normal;


            outInt.uv = v0.uv * weights.x +
                v1.uv * weights.y +
                v2.uv * weights.z;
            outInt.normal =
                Math::normalize(n0) * weights.x +
                Math::normalize(n1) * weights.y +
                Math::normalize(n2) * weights.z;
            outInt.position = ray.origin + ray.dir * tmin;
            //outInt.position = v0.position * weights.x +
            //                  v1.position * weights.y +
            //                  v2.position * weights.z;


        }
        //__syncwarp();
        return hit;
    }



    __device__ void RayHitBox(const Ray& ray, const AABB& box, Math::vec3f& outNorm, float& outT) {


        float   origx, origy, origz;    // Ray origin.
        float   dirx, diry, dirz;       // Ray direction.
        //float   tmin;                   // t-value from which the ray starts. Usually 0.
        float   idirx, idiry, idirz;    // 1 / dir
        float   oodx, oody, oodz;       // orig / dir


        origx = ray.origin.x;
        origy = ray.origin.y;
        origz = ray.origin.z;
        dirx = ray.dir.x;
        diry = ray.dir.y;
        dirz = ray.dir.z;

        // ooeps is very small number, used instead of raydir xyz component when that component is near zero
        float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
        idirx = 1.0f / (fabsf(ray.dir.x) > ooeps ? ray.dir.x : copysignf(ooeps, ray.dir.x)); // inverse ray direction
        idiry = 1.0f / (fabsf(ray.dir.y) > ooeps ? ray.dir.y : copysignf(ooeps, ray.dir.y)); // inverse ray direction
        idirz = 1.0f / (fabsf(ray.dir.z) > ooeps ? ray.dir.z : copysignf(ooeps, ray.dir.z)); // inverse ray direction
        oodx = origx * idirx;  // ray origin / ray direction
        oody = origy * idiry;  // ray origin / ray direction
        oodz = origz * idirz;  // ray origin / ray direction


        float lox = box.min.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
        float hix = box.max.x * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
        float loy = box.min.y * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
        float hiy = box.max.y * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
        float loz = box.min.z * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
        float hiz = box.max.z * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
        
        lox = fminf(lox, hix);
        loy = fminf(loy, hiy);
        loz = fminf(loz, hiz);

        int axis = 0;
        if (lox > loy) axis = (lox > loz) ? 0 : 2;
        else axis = (loy > loz)?1:2;
        outNorm = Math::vec3f(0);
        outNorm[axis] = -copysignf(1,ray.dir[axis]);
        outT = fmaxf(fmaxf(lox, loy),loz);
    }

    struct PRD {
        LCG<>* randGen;
        unsigned ttl;/*Time to live, that is how many light bounces left*/
    };

    class RTProgram {

    public:
        SceneInfoHeader* scene;
        PathTraceSettings* settings;

        __both__ RTProgram(SceneInfoHeader* scene, PathTraceSettings* settings, bool refractive = true)
            :scene(scene), settings(settings){}

        __device__ Math::vec3f GenRay(const Math::vec2i& rayID, std::uint32_t mask) {

            float num_x = settings->viewportWidth;
            float num_y = settings->viewportHeight;


            unsigned local_seed = settings->viewportWidth * rayID.y + rayID.x;
            //local_seed += settings->seed;

            LCG<> randGen{};
            randGen.init(local_seed, settings->frameID);

            PRD prd{};
            prd.ttl = settings->maxDepth;
            prd.randGen = &randGen;

            auto& cam = settings->cam;

            

            float edgeLen = min(num_x, num_y);

            float fovScale = std::atan(cam.fov / 2);
            float pixelSize = 1.0 / edgeLen * fovScale * 2;

            float w = num_x * pixelSize;
            float h = num_y * pixelSize;


            float origin_x = -w / 2;
            float origin_y = -h / 2;

            float offx = randGen.NextFloat() * pixelSize;
            float offy = randGen.NextFloat() * pixelSize;

            float u = (rayID.x * pixelSize + origin_x + offx); //convert to (-1,1)
            float v = (rayID.y * pixelSize + origin_y + offy);



            auto rayDir = cam.front + cam.right * u + cam.up * v;

            rayDir = Math::normalize(rayDir);

            float r = rayDir.x;
            float g = rayDir.y;
            float b = rayDir.z;

            Ray ray{
                cam.pos, rayDir
            };
            //if(useRefractive)
                return this->LaunchRayRecur(ray, &prd, mask);
            //else
                //return this->LaunchRay(ray, &prd, mask);

        }


        __device__ Math::vec3f LaunchRayRecur(const Ray& ray, PRD* prd, std::uint32_t mask) {
            //auto mask = __activemask();
            int ttl = prd->ttl;
            Math::vec3f currColor(0, 0, 0);
            //
            auto rayMask = __ballot_sync(mask, ttl > 0);
            //
            if(ttl > 0){
                Intersection hit;
                bool hasIntc = TraceRay(ray, hit, mask);

                auto launchMask = __ballot_sync(mask, hasIntc);

                if (hasIntc) {
                    const auto& mesh = *(const MeshInfoHeader*)hit.object;
                    const auto& mat = mesh.material;
                    auto surfaceNorm = hit.frontFacing? hit.normal : - hit.normal;

                    auto norm = GGXRandDirInsideCone(*(prd->randGen), surfaceNorm, mat.roughness);

                    auto dir = Math::reflect(ray.dir, norm);

                    Ray out_ray = {
                        hit.position + surfaceNorm * 1e-6f,
                        dir//raySelDir
                    };


                    PRD thisprd{};
                    thisprd.ttl = ttl - 1;
                    thisprd.randGen = prd->randGen;

                    auto c = LaunchRayRecur(out_ray, &thisprd, launchMask);
                    c = Math::clamp(c,Math::vec3f(0),Math::vec3f(Math::infty()));

                        //Refraction index(incident)
                        //float n1 = prd->ni;
                        float n1 = 1.f;
                        //Refraction index(refraction)
                        float n2 = mat.n;
                        n2 += FLT_EPSILON;
                        //Outbound ray, fetch prev n

                        if (!hit.frontFacing) {
                            auto t = n1; n1 = n2; n2 = t;
                        }
                        //Actual normal
                        //auto sampleNorm = Math::normalize(raySelDir - ray.dir);

                        //Schlick's approximation of Fresnel equation
                        //r(x) = r0 + (1 - r0)(1 - cos(x))^5
                        //r0 = ((n1 - n2)/(n1 + n2))^2
                        float r0 = (n1 - n2) / (n1 + n2); r0 *= r0;

                        //float rr = r0 + (1 - r0) * (1 - Math::dot(-ray.dir, sampleNorm))^5;
                        float rr = (1 - Math::dot(-ray.dir, norm));
                        rr = powf(rr, 5);
                        rr = r0 + (1 - r0) * rr;

                        rr = __saturatef(rr);

                        //Opt-in metallic param
                        rr = mat.metallic * 1.f + (1 - mat.metallic) * rr;

                        float rt = 1 - rr;

                        auto reflection_absorb = mat.metallic * mat.color + (1 - mat.metallic) * Math::vec3f(1.f);
                        currColor = reflection_absorb * c * rr;

                        if(mat.opacity < 1.f){
                            auto front = (ray.dir + dir) / 2;
                            auto down = ray.dir - front;
                            float down_len = Math::length(down);
                            down = down_len < 1e-10?Math::vec3f(1,0,0):(down/down_len);
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

                            out_ray.dir = refr_dir;
                            out_ray.origin = hit.position - surfaceNorm * 1e-6f;

                        }else{
                            out_ray.dir = RandDirInsideCone(
                                *(prd->randGen), surfaceNorm, 0.49);
                            out_ray.origin = hit.position + surfaceNorm * 1e-6f;

                        }
                        __syncwarp(launchMask);
                        c = LaunchRayRecur(out_ray, &thisprd, launchMask);
                        currColor += mat.color * c * rt * ((mat.opacity == 1.f)? 1.f : ( 1 - mat.opacity));
                    
                    currColor += mat.emissiveColor * mat.emissiveStrength;

                }else{
                    currColor = OnMiss(ray, nullptr);
                }

                

            }

            __syncwarp(mask);

            return currColor;
        }

        __device__ bool TraceRay(const Ray& ray, Intersection& hit, std::uint32_t mask) {
            //auto mask = __activemask();

            BVHNode* traversalStack[BVH_DEPTH + 1];

            float   origx, origy, origz;    // Ray origin.
            float   dirx, diry, dirz;       // Ray direction.
            //float   tmin;                   // t-value from which the ray starts. Usually 0.
            float   idirx, idiry, idirz;    // 1 / dir
            float   oodx, oody, oodz;       // orig / dir

            BVHNode** stackPtr;
            BVHNode*  leafAddr;
            BVHNode*  nodeAddr;
            int       hitIndex;
            float     hitT;
            
            origx = ray.origin.x;
            origy = ray.origin.y;
            origz = ray.origin.z;
            dirx = ray.dir.x;
            diry = ray.dir.y;
            dirz = ray.dir.z;

            // ooeps is very small number, used instead of raydir xyz component when that component is near zero
            float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
            idirx = 1.0f / (fabsf(ray.dir.x) > ooeps ? ray.dir.x : copysignf(ooeps, ray.dir.x)); // inverse ray direction
            idiry = 1.0f / (fabsf(ray.dir.y) > ooeps ? ray.dir.y : copysignf(ooeps, ray.dir.y)); // inverse ray direction
            idirz = 1.0f / (fabsf(ray.dir.z) > ooeps ? ray.dir.z : copysignf(ooeps, ray.dir.z)); // inverse ray direction
            oodx = origx * idirx;  // ray origin / ray direction
            oody = origy * idiry;  // ray origin / ray direction
            oodz = origz * idirz;  // ray origin / ray direction

            
            //Intersection hit;
            float tmin = Math::infty();
            bool hasIntc = false;
            auto payload = scene->GetPayload();

            //auto mask = __activemask();

            for (unsigned i = 0; i < scene->meshCnt; i++){

                auto pmesh = scene->GetPayload()[i];
                auto& mesh = *(MeshInfoHeader*)pmesh;
                const Math::mat4x4f& tran = mesh.transform;

                traversalStack[0] = nullptr; // Bottom-most entry. 0x76543210 is 1985229328 in decimal
                stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
                leafAddr = nullptr;   // No postponed leaf.
                nodeAddr = mesh.bvh;   // Start from the root.
                hitIndex = -1;  // No triangle intersected so far.
                hitT = Math::infty();//raydir.w;

                //pass leaf node directly
                if(nodeAddr->isLeaf) leafAddr = nodeAddr;
                //AABB boxr = nodeAddr->box;//.Transformed(tran);
                //
                //float rlox = boxr.min.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
                //float rhix = boxr.max.x * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
                //float rloy = boxr.min.y * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
                //float rhiy = boxr.max.y * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
                //float rloz = boxr.min.z * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
                //float rhiz = boxr.max.z * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
                //float rmin = spanBeginKepler(rlox, rhix, rloy, rhiy, rloz, rhiz, 0/*tmin*/); // Tesla does max4(min, min, min, tmin)
                //float rmax = spanEndKepler(rlox, rhix, rloy, rhiy, rloz, rhiz, hitT); // Tesla does min4(max, max, max, tmax)
                

                float ray_tmax = 1e20;
                //bool traverseRoot = (rmin <= rmax); // && (c0min >= tmin) && (c0min <= ray_tmax);
                //if(!traverseRoot)
                //    nodeAddr = nullptr;
                
                //__syncwarp(mask);
                auto launchgroupMask = mask;
                int stillWorking = 1;
                while (true) {
                    // Traverse internal nodes until all SIMD lanes have found a leaf.

                    //Flag out exited threads
                    stillWorking = nodeAddr != nullptr;
                    launchgroupMask = __ballot_sync(launchgroupMask, stillWorking);
                    if (!stillWorking) break;

                    auto workgroupMask = launchgroupMask;// __ballot_sync(launchgroupMask, stillWorking);
                    bool searchingLeaf = true; // flag required to increase efficiency of threads in warp
                    while (true)
                    {
                        //Work with longest worker in sync as long as possible
                        //Flag out already filled threads
                        stillWorking = nodeAddr != nullptr && !nodeAddr->isLeaf;
                        workgroupMask = __ballot_sync(workgroupMask, stillWorking);

                        //Pending queue full, can't proceed anymore
                        if(!stillWorking) break;

                        BVHNode* c0 = nodeAddr->lchild;
                        BVHNode* c1 = nodeAddr->rchild;

                        const AABB& box0 = c0->box;//.Transformed(tran);
                        const AABB& box1 = c1->box;//.Transformed(tran);

                        //AABB box0 = c0->box;
                        //AABB box1 = c1->box;

                        
                        // compute ray intersections with BVH node bounding box

                        float c0lox = box0.min.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
                        float c0hix = box0.max.x * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
                        float c0loy = box0.min.y * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
                        float c0hiy = box0.max.y * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
                        float c0loz = box0.min.z * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
                        float c0hiz = box0.max.z * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
                        float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0); // Tesla does max4(min, min, min, tmin)
                        float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
                        float c1lox = box1.min.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
                        float c1hix = box1.max.x * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
                        float c1loy = box1.min.y * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
                        float c1hiy = box1.max.y * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
                        float c1loz = box1.min.z * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
                        float c1hiz = box1.max.z * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
                        float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
                        float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

                        //float ray_tmax = 1e20;
                        bool traverseChild0 = (c0min <= c0max); // && (c0min >= tmin) && (c0min <= ray_tmax);
                        bool traverseChild1 = (c1min <= c1max); // && (c1min >= tmin) && (c1min <= ray_tmax);

                        if (!traverseChild0 && !traverseChild1)
                        {
                            nodeAddr = *stackPtr; // fetch next node by popping stack
                            stackPtr --; // popping decrements stack by 4 bytes (because stackPtr is a pointer to char) 
                        }

                        // Otherwise => fetch child pointers.

                        else  // one or both children intersected
                        {
                            //int2 cnodes = *(int2*)&ptr[3];
                            // set nodeAddr equal to intersected childnode (first childnode when both children are intersected)
                            nodeAddr = (traverseChild0) ? c0 : c1;

                            // Both children were intersected => push the farther one on the stack.

                            if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
                            {
                                if (c1min < c0min)
                                    swap2(nodeAddr, c1);
                                stackPtr ++;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
                                *stackPtr = c1; // push furthest node on the stack
                            }
                        }

                        // First leaf => postpone and continue traversal.
                        // leafnodes have a negative index to distinguish them from inner nodes
                        // if nodeAddr less than 0 -> nodeAddr is a leaf
                        if (nodeAddr != nullptr && nodeAddr->isLeaf && leafAddr == nullptr)  // if leafAddr >= 0 -> no leaf found yet (first leaf)
                        {
                            searchingLeaf = false; // required for warp efficiency
                            leafAddr = nodeAddr;

                            nodeAddr = *stackPtr;  // pops next node from stack
                            stackPtr --;  // decrement by 4 bytes (stackPtr is a pointer to char)
                        }

                        // All SIMD lanes have found a leaf => process them.
                        // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
                        // tried everything with CUDA 4.2 but always got several redundant instructions.

                        // if (!searchingLeaf){ break;  }  
                        //auto currMask = __activemask();
                        if (!__any_sync(workgroupMask, searchingLeaf)) break; // "__any" keyword: if none of the threads is searching a leaf, in other words
                        // if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

                        // if(!__any(leafAddr >= 0))   /// als leafAddr in PTX code >= 0, dan is het geen echt leafNode   
                        //    break;

                    }

                    __syncwarp(launchgroupMask);
                    ///////////////////////////////////////////
                    /// TRIANGLE INTERSECTION
                    //////////////////////////////////////

                    // Process postponed leaf nodes.

                    while (leafAddr != nullptr && leafAddr->isLeaf)  /// if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode)
                    {
                        // Intersect the ray against each triangle using Sven Woop's algorithm.
                        // Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
                        // must be transformed to "unit triangle space", before testing for intersection

                        
                    #if VISUALIZE_BVH
                        hasIntc = true;
                        Math::vec3f norm; float thisT;
                        RayHitBox(ray, leafAddr->box, norm, thisT);
                        if (thisT < tmin) {
                            tmin = thisT;

                            hit.normal = norm;
                            hit.t = thisT;
                            hit.frontFacing = true;
                            hit.position = ray.origin + ray.dir * thisT;
                            hit.object = &mesh;
                        }
                    #else
                        Intersection thisHit;
                        thisHit.object = &mesh;
                        
                        auto triangleArr = (std::uint32_t*)(leafAddr + 1);
                        auto triangleCnt = leafAddr->triangleCnt;

                        if (NearestIntersectSubset(mesh, ray, triangleArr, triangleCnt, thisHit)) {
                            hasIntc = true;
                        
                            if (thisHit.t < tmin){
                                tmin = thisHit.t;
                        
                                hit = thisHit;
                            }
                        }
                    #endif
                        //hit.t = hitT;
                        //hit.object = &mesh;
                        //hit.position = ray.dir * hitT + ray.origin;

                        // Another leaf was postponed => process it as well.

                        leafAddr = nodeAddr;
                        if (nodeAddr != nullptr && nodeAddr->isLeaf)    // nodeAddr is an actual leaf when < 0
                        {
                            nodeAddr = *stackPtr;  // pop stack
                            stackPtr --;               // decrement with 4 bytes to get the next int (stackPtr is char*)
                        }
                    }

                    //__syncwarp(__activemask());


                    leafAddr = nullptr;

                }

                //__syncwarp(__activemask());
                __syncwarp(mask);

            }
            return hasIntc;

        }

        __device__ Math::vec3f OnMiss(const Ray& ray, PRD* prd) {
            //return { 1,0.3,0.5 };

            if(scene->envMap == nullptr){

                const auto& skycolor = scene->skyLight.color;
                const auto& sundir = scene->skyLight.dir;
                float intensity = scene->skyLight.intensity;

                float ang = (1 - Math::dot(ray.dir, sundir)) / 2;

                return skycolor * intensity * powf(ang ,10);
            }
            else {
                const auto& dim = scene->envMapDim;
                const auto base = scene->envMap;

                float zenith = acos(ray.dir.y);
                float azimuth = atan2(ray.dir.x, ray.dir.z);

                float v = zenith / M_PI;
                float u = azimuth / (M_PI * 2) + 0.5;

                //return {0,v,0};

                u *= dim.x; v *= dim.y;

                //int u0 = floor(u), u1 = ceil(u);
                //int v0 = floor(v), v1 = ceil(v);
                int u0 = u, u1 = u;
                int v0 = v, v1 = v;
                //u0 %= dim.x; u1 %= dim.x;
                //v0 %= dim.y; v1 %= dim.y;

                unsigned a00 = v0 * dim.x + u0;
                unsigned a01 = v1 * dim.x + u0;
                unsigned a10 = v0 * dim.x + u1;
                unsigned a11 = v1 * dim.x + u1;

                Math::vec3f p00(base[a00 * 3], base[a00 * 3 + 1], base[a00 * 3 + 2]);
                Math::vec3f p01(base[a01 * 3], base[a01 * 3 + 1], base[a01 * 3 + 2]);
                Math::vec3f p10(base[a10 * 3], base[a10 * 3 + 1], base[a10 * 3 + 2]);
                Math::vec3f p11(base[a11 * 3], base[a11 * 3 + 1], base[a11 * 3 + 2]);

                float du = u - u0, dv = v - v0;
                auto p0 = (dv) * p01 + ( 1 - dv ) * p00;
                auto p1 = (dv) * p11 + ( 1 - dv ) * p10;
                auto p = du * p1 + (1 - du) * p0;
                return p;
            }
        }

        __device__ Math::vec3f Test(Math::vec2f& id) {

            float num_x = settings->viewportWidth;
            float num_y = settings->viewportHeight;
            return { id.x / num_x, id.y / num_y, 0.8 };
        }

    };

    

    __global__ void
    __launch_bounds__(BATCH_W*BATCH_H/*, minBlocksPerMultiprocessor*/)
    GenRay(
        RTProgram program,
        //SceneInfoHeader* scene, PathTraceSettings* settings,
        float* accBuffer,
        char* fb
    ) {
        int w = program.settings->viewportWidth;
        int h = program.settings->viewportHeight;
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        bool isThreadInRange = !(i < 0 || i >= w || j < 0 || j >= h); 

        auto warpMask = __ballot_sync(__activemask(), isThreadInRange);

        if(!isThreadInRange) return;
        float weight = 1.0 / (program.settings->frameID + 1.0);
        //float weight = 1.0;
        
        //do the launching inside viewport
        Math::vec3f rc = program.GenRay({ i,j }, warpMask);
        //Math::vec3f c = program.Test({i,j});
        float r = rc.x;//(rc.x >= 0)? rc.x : 0;
        float g = rc.y;//(rc.y >= 0)? rc.y : 0;
        float b = rc.z;//(rc.z >= 0)? rc.z : 0;
        Math::vec3f c{r,g,b};
        //Gamma correction
        
        
        //Assume buffer format RGBA and row priori storage
        auto cbuf = (float*)accBuffer;
        int addr = (j * w + i) * 4;
        
        r = cbuf[addr] * (1 - weight) + c.x * weight;
        g = cbuf[addr + 1] * (1 - weight) + c.y * weight;
        b = cbuf[addr + 2] * (1 - weight) + c.z * weight;

        //float r = c.x;
        //float g = c.y;
        //float b = c.z;

        cbuf[addr] = r;
        cbuf[addr + 1] = g;
        cbuf[addr + 2] = b;
        cbuf[addr + 3] = 1;

        c = {
            __saturatef(r),
            __saturatef(g),
            __saturatef(b)
        };
        c = c / (c + 1.0f);
        c = Math::pow(c, Math::vec3f(1.0f / 2.2f));
        
        fb[addr] =     c.x * 255;
        fb[addr + 1] = c.y * 255;
        fb[addr + 2] = c.z * 255;
        fb[addr + 3] = 255;


    }


    PathTracer::PathTracer() {
        cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192);
    }

    void PathTracer::Trace(
        CUDAScene& scn, TypedBuffer<PathTraceSettings>& settings,
        float* accBuffer, char* buffer
    ){
        auto stdata = (PathTraceSettings*)settings.cpu_data();

        auto w = stdata->viewportWidth;
        auto h = stdata->viewportHeight;


        const unsigned batchW = BATCH_W, batchH = BATCH_H;
        const unsigned blockW = (w + batchW - 1) / batchW;
        const unsigned blockH = (h + batchH - 1) / batchH;

        uint3 blkPerGrid{ blockW, blockH, 1 };
        //uint3 blkPerGrid{ 2, 2, 1 };
        uint3 threadPerBlk{batchW, batchH, 1};

        auto scene_gpu = (SceneInfoHeader*)scn.GetGPUData();
        auto settings_gpu = (PathTraceSettings*)settings.gpu_data();
        //auto cache_gpu = (std::uint8_t*)_rayCache->gpu_data();
        RTProgram prog{ scene_gpu, settings_gpu};

        GenRay << <blkPerGrid, threadPerBlk >> > (prog, accBuffer, buffer);

        
    }

}

