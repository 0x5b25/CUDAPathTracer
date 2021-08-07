#pragma once

#include <vector>
#include <functional>

#include "Math/Vec.hpp"
#include "Math/Matrix.hpp"
#include "Math/MathCommon.h"
#include "CUDACommon.h"

#define BVH_DEPTH 32
#define BVH_MIN_TRICNT 4

namespace CUDATracer {

    class Mesh;
    
    struct Camera
    {
        Math::vec3f pos;
        Math::vec3f front;
        Math::vec3f up;
        Math::vec3f right;
        float fov;
    };
    
    struct Vertex {
        Math::vec3f position;
        Math::vec3f normal;
        Math::vec2f uv;
    };


    struct Ray {
        Math::vec3f origin;
        Math::vec3f dir;
    };

    struct AABB {
        Math::vec3f min, max;

        __both__ AABB() :min(Math::infty()), max(-Math::infty()) {}
        __both__ AABB(const Math::vec3f& min, const Math::vec3f& max)
            :min(min), max(max){}
        __both__ AABB(const AABB&) = default;
        __both__ AABB(AABB&&) = default;
        __both__ AABB& operator=(const AABB&) = default;
        __both__ AABB& operator=(AABB&&) = default;


        template<typename _ListTy>
        static AABB FromPoints(const _ListTy& list) {
            AABB _box{(float)Math::inf, (float)Math::neg_inf };

            for (auto& e : list) {
                auto p = e;
                if (p.x < _box.min.x) _box.min.x = p.x;
                if (p.x > _box.max.x) _box.max.x = p.x;

                if (p.y < _box.min.y) _box.min.y = p.y;
                if (p.y > _box.max.y) _box.max.y = p.y;

                if (p.z < _box.min.z) _box.min.z = p.z;
                if (p.z > _box.max.z) _box.max.z = p.z;
            }

            return _box;
        }

        template <typename _ListTy, typename _GenTy>
        static AABB FromList(const _ListTy& list, _GenTy gen) {
            AABB _box((float)Math::inf, (float)Math::neg_inf);

            for (auto& e : list) {
                auto p = gen(e);
                if (p.x < _box.min.x) _box.min.x = p.x;
                if (p.x > _box.max.x) _box.max.x = p.x;

                if (p.y < _box.min.y) _box.min.y = p.y;
                if (p.y > _box.max.y) _box.max.y = p.y;

                if (p.z < _box.min.z) _box.min.z = p.z;
                if (p.z > _box.max.z) _box.max.z = p.z;
            }

            return _box;
        }

        template <typename _GenTy>
        static AABB Generate(unsigned count, _GenTy gen) {
            AABB _box{ (float)Math::inf, (float)Math::neg_inf };

            for(unsigned i = 0; i < count; i++){
                auto p = gen(i);
                if (p.x < _box.min.x) _box.min.x = p.x;
                if (p.x > _box.max.x) _box.max.x = p.x;

                if (p.y < _box.min.y) _box.min.y = p.y;
                if (p.y > _box.max.y) _box.max.y = p.y;

                if (p.z < _box.min.z) _box.min.z = p.z;
                if (p.z > _box.max.z) _box.max.z = p.z;
            }

            return _box;
        }
#ifdef __CUDACC__
    #define invoke(x) ::x
#else
    #define invoke(x) std::x
#endif
        /*Union operation*/
        __both__ AABB operator+(const AABB& b)const{
            const AABB& a = *this;
            AABB _aabb;
            _aabb.min.x = invoke(min)(a.min.x, b.min.x);
            _aabb.max.x = invoke(max)(a.max.x, b.max.x);
            _aabb.min.y = invoke(min)(a.min.y, b.min.y);
            _aabb.max.y = invoke(max)(a.max.y, b.max.y);
            _aabb.min.z = invoke(min)(a.min.z, b.min.z);
            _aabb.max.z = invoke(max)(a.max.z, b.max.z);

            return _aabb;
        }

        /*Overlap operation*/
        __both__ AABB operator-(const AABB& b)const {
            const AABB& a = *this;
            AABB _aabb;
            _aabb.min.x = invoke(max)(a.min.x, b.min.x);
            _aabb.max.x = invoke(min)(a.max.x, b.max.x);
            _aabb.min.y = invoke(max)(a.min.y, b.min.y);
            _aabb.max.y = invoke(min)(a.max.y, b.max.y);
            _aabb.min.z = invoke(max)(a.min.z, b.min.z);
            _aabb.max.z = invoke(min)(a.max.z, b.max.z);

            return _aabb;
        }

        /*Is overlapping*/
        __both__ bool DoesOverlap(const AABB& b)const {
            AABB _aabb = *this-b;// AABB::operator-(a,b);
            return _aabb.min.x < _aabb.max.x 
                && _aabb.min.y < _aabb.max.y
                && _aabb.min.z < _aabb.max.z
            ;
        }

        __both__ const AABB Transformed(const Math::mat4x4f& transform) const {
            Math::vec3f v0{ min.x, min.y, min.z/*, 1.0*/ };
            Math::vec3f v1{ min.x, min.y, max.z/*, 1.0*/ };
            Math::vec3f v2{ max.x, min.y, max.z/*, 1.0*/ };
            Math::vec3f v3{ max.x, min.y, min.z/*, 1.0*/ };
            Math::vec3f v4{ min.x, max.y, min.z/*, 1.0*/ };
            Math::vec3f v5{ min.x, max.y, max.z/*, 1.0*/ };
            Math::vec3f v6{ max.x, max.y, max.z/*, 1.0*/ };
            Math::vec3f v7{ max.x, max.y, min.z/*, 1.0*/ };

            Math::vec3f vt[] = {
                Math::MatMul(transform, v0, 1.f), Math::MatMul(transform, v1, 1.f),
                Math::MatMul(transform, v2, 1.f), Math::MatMul(transform, v3, 1.f),
                Math::MatMul(transform, v4, 1.f), Math::MatMul(transform, v5, 1.f),
                Math::MatMul(transform, v6, 1.f), Math::MatMul(transform, v7, 1.f),
            };

            AABB res;

            Math::vec3f& min = res.min, &max = res.max;

            for (auto& v : vt) {
                if (v.x < min.x) min.x = v.x;
                if (v.x > max.x) max.x = v.x;

                if (v.y < min.y) min.y = v.y;
                if (v.y > max.y) max.y = v.y;

                if (v.z < min.z) min.z = v.z;
                if (v.z > max.z) max.z = v.z;
            }

            return res;
        }
    
        __both__ float AreaX()const { return (max.y - min.y) * (max.z - min.z); }
        __both__ float AreaY()const { return (max.x - min.x) * (max.z - min.z); }
        __both__ float AreaZ()const { return (max.x - min.x) * (max.y - min.y); }

        inline    void            grow(const Math::vec3f& pt) { min = Math::max(min, pt); max = Math::min(max, pt); } // grows bounds to include 3d point pt
        inline    void            grow(const AABB& aabb) { grow(aabb.min); grow(aabb.max); }
        inline    void            intersect(const AABB& aabb) { min = Math::max(min, aabb.min); max = Math::min(max, aabb.max); } /// box formed by intersection of 2 AABB boxes
        inline    float           volume(void) const { if (!valid()) return 0.0f; return Math::volume(max - min); } /// volume = AABB side along X-axis * side along Y * side along Z
        inline    float           area(void) const { if (!valid()) return 0.0f; auto d = max - min; return (d.x * d.y + d.y * d.z + d.z * d.x) * 2.0f; }
        inline    bool            valid(void) const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }
        inline    Math::vec3f           midPoint(void) const { return (min + max) * 0.5f; } // AABB centroid or midpoint
        

    };


    struct Intersection {
        Math::vec3f position;
        //Math::vec3f weights;
        Math::vec3f normal;
        Math::vec2f uv;
        bool frontFacing;
        float t;

        const void* object;
        //unsigned primitiveID;
    };

    struct SurfaceMaterial
    {
        Math::vec3f color;
        float roughness;
        float metallic;
        float opacity;
        /*Refraction index relative to air(1)*/
        float n;
        bool translucent;
    };

    struct SkyLight {
        Math::vec3f color;
        Math::vec3f dir;
        float intensity;
    };


    struct BVHNode {
        AABB box;
        std::uint32_t isLeaf;
        std::uint32_t triangleCnt;
        //aligned to 8-byte spaces
        BVHNode* lchild, * rchild;

        std::size_t TotalSize() const {
            auto baseSize = sizeof(BVHNode);

            //Align to 8 byte pointers:
            if(!isLeaf) return baseSize;
            auto listSize = triangleCnt * sizeof(std::uint32_t) * 3;
            auto alignedListSize = ((listSize + 7) / 8) * 8;

            return baseSize + alignedListSize;
        }
    };

    void DeleteBVH(BVHNode* node);
    BVHNode* BuildBVH(const Mesh& mesh, unsigned depth = BVH_DEPTH, unsigned minLeafTriCnt = BVH_MIN_TRICNT);

    class BVH {
    public:
        BVHNode* root;

        BVH(const Mesh& mesh, unsigned depth = BVH_DEPTH, unsigned minLeafTriCnt = BVH_MIN_TRICNT)
            : root(BuildBVH(mesh, depth, minLeafTriCnt))
        {

        }

        ~BVH() {
            DeleteBVH(root);
        }
    };

    //Triangle primitive: counter-clockwise faces up
    class Mesh{
        std::vector<Vertex> _vert;
        std::vector<std::uint32_t> _ind;
        AABB _aabb;

        std::unique_ptr<BVH> _bvh;

        SurfaceMaterial _mat;
    public:
        Math::mat4x4f transform;

        Mesh(const std::vector<Vertex>& vert, const std::vector<std::uint32_t>& ind,
            const SurfaceMaterial& mat
        )
            :_vert(vert), _ind(ind), _mat(mat), transform(1) {

            _aabb = AABB::FromList(vert, [&](const Vertex& v) {
                return v.position;
            });

            _bvh = std::make_unique<BVH>(*this);
        }

        Mesh(Mesh&& other) = default;
        Mesh& operator=(Mesh && other) = default;

        ~Mesh()
        {
        }

        const AABB GetAABB() const { return _aabb; }
        const BVHNode* GetBVH() const { return _bvh->root; }
        
        const SurfaceMaterial& GetMat() const { return _mat; }
        const std::vector<Vertex>& GetVertices() const {return _vert;}
        const std::vector<std::uint32_t>& GetIndices() const {return _ind;}


    };

    inline BVHNode* BuildBVH(const Mesh& mesh, unsigned depth, unsigned minLeafTriCnt) {
        auto& verts = mesh.GetVertices();
        auto& inds = mesh.GetIndices();
        std::vector<std::uint32_t> triangleIDs(inds.size() / 3);
        for (int i = 0; i < triangleIDs.size(); i++) {
            triangleIDs[i] = i;
        }

        std::vector<Math::vec3f> centers; centers.reserve(triangleIDs.size());
        for (auto triangleID : triangleIDs) {
            auto vertID = triangleID * 3;
            const auto& v0 = verts[inds[vertID]];
            const auto& v1 = verts[inds[vertID + 1]];
            const auto& v2 = verts[inds[vertID + 2]];
            auto center = v0.position + v1.position + v2.position;
            centers.push_back(center / 3);
        }

        std::function<BVHNode*(std::vector<std::uint32_t>&, int)> _Build =
        [&](std::vector<std::uint32_t>& triIdx, auto ttl) {
            BVHNode* node;

            //Build bounding box
            auto box = AABB::Generate(triIdx.size() * 3, [&](auto idx) {
                int addr = idx / 3;
                int off = idx - addr * 3;
                auto triangelID = triIdx[addr];
                auto vertID = triangelID *3+ off;
                auto vertAddr = inds[vertID];
                const Vertex& vert = verts[vertAddr];
                return vert.position;
            });

            if (ttl <= 0 || triIdx.size() <= std::max(1U,minLeafTriCnt) ) {
                auto listSize = triIdx.size() * sizeof(std::uint32_t) * 3;
                auto totalSize = sizeof(BVHNode) + listSize;
                node = (BVHNode*)malloc(totalSize);
                node->isLeaf = true;
                auto buffer = (std::uint32_t*)(node + 1);

                for (int i = 0; i < triIdx.size(); i++) {
                    auto tid =  triIdx[i];
                    auto taddr = tid * 3;
                    auto addr = i * 3;
                    buffer[addr] = inds[taddr];
                    buffer[addr + 1] = inds[taddr + 1];
                    buffer[addr + 2] = inds[taddr + 2];

                }

                //memcpy(buffer, triIdx.data(), listSize);
            }
            else {
                auto totalSize = sizeof(BVHNode);
                node = (BVHNode*)malloc(totalSize);
                node->isLeaf = false;
                auto triCnt = triIdx.size();
                //calculate center point
                
                //Measure center point distributions
                auto cbox = AABB::FromList(triIdx, [&](int id){
                    return centers[id];
                });
                float len_x = cbox.max.x - cbox.min.x;
                float len_y = cbox.max.y - cbox.min.y;
                float len_z = cbox.max.z - cbox.min.z;
                //Max axis
                int ax = 0;
                if (len_x >= len_y) {
                    if(len_x >= len_z) ax = 0; else ax = 2;
                }
                else {
                    if(len_y >= len_z) ax = 1; else ax = 2;
                }
                auto pivot = triIdx.begin() + triCnt / 2;
                //
                std::nth_element(
                    triIdx.begin(), pivot, triIdx.end(),[&](unsigned a, unsigned b){
                    //less than
                    return centers[a][ax] < centers[b][ax];
                });

                std::vector<std::uint32_t> ltri(triIdx.begin(), triIdx.begin() + triCnt / 2);
                std::vector<std::uint32_t> rtri(triIdx.begin() + triCnt / 2, triIdx.end());

                auto lch = _Build(ltri, ttl - 1);
                auto rch = _Build(rtri, ttl - 1);

                node->lchild = lch;
                node->rchild = rch;
            }

            node->box = box;
            node->triangleCnt = triIdx.size();

            return node;
        };

        

        return _Build(triangleIDs, depth);
    }

    inline void DeleteBVH(BVHNode* node) {
        if(node == nullptr) return;
        if (!node->isLeaf) {
            DeleteBVH(node->lchild);
            DeleteBVH(node->rchild);
        }
        free(node);
    }
    



    struct Scene {
        SkyLight skyLight;
        std::vector<float> envMap;
        Math::vec2ui envMapDim;

        std::vector<Mesh> objects;
    };

}
