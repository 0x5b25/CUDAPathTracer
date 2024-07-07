#pragma once

#include "CUDATracer.hpp"

namespace CUDATracer {

    /*
        +--------+----------+---------+
        | Header | Vertices | Indices |
        +--------+----------+---------+
    */
    struct MeshInfoHeader {
        AABB aabb;
        BVHNode* bvh;
        SurfaceMaterial material;
        Math::mat4x4f transform;
        std::uint32_t vertexCnt, indicesCnt;

        __both__  inline unsigned TotalSize() const {
            return sizeof(MeshInfoHeader)
                + sizeof(Vertex) * vertexCnt
                + sizeof(std::uint32_t) * indicesCnt
                ;
        }

        __both__ inline Vertex* GetVertices() { return(Vertex*)(this + 1); }
        __both__ inline const Vertex* GetVertices() const { return(const Vertex*)(this + 1); }

        __both__ inline std::uint32_t* GetIndices() {
            return(std::uint32_t*)((char*)this + sizeof(Vertex) * vertexCnt + sizeof(MeshInfoHeader));
        }

        __both__ inline const std::uint32_t* GetIndices() const {
            return(const std::uint32_t*)((const char*)this + sizeof(Vertex) * vertexCnt + sizeof(MeshInfoHeader));
        }

    };

    /*
        +--------+---------------------+
        | Header | MeshInfo* [meshCnt] |
        +--------+---------------------+
    */
    struct SceneInfoHeader {
        SkyLight skyLight;
        float* envMap;
        Math::vec2ui envMapDim;
        unsigned meshCnt;

        __both__ MeshInfoHeader** GetPayload() { return (MeshInfoHeader**)(this + 1); }
        __both__ const MeshInfoHeader** GetPayload() const
        {
            return (const MeshInfoHeader**)(this + 1);
        }

        __both__ MeshInfoHeader*& operator[](unsigned index) {
            assert(index < meshCnt);
            return GetPayload()[index];
        }

        __both__ const MeshInfoHeader* operator[](unsigned index) const {
            assert(index < meshCnt);
            return GetPayload()[index];
        }
    };

    template<typename T>
    T* as(void* ptr) { return (T*)ptr; }

    template<typename T>
    const T* as(const void* ptr) { return (const T*)ptr; }

    class CUDATraceable : public ITraceable {

    public:

        DISABLE_COPY_AND_ASSIGN(CUDATraceable);

        std::unique_ptr<CUDABuffer> _SceneInfo;
        std::unique_ptr<CUDABuffer> _EnvMap;
        std::vector<std::unique_ptr<CUDABuffer>> _MeshInfoArr;

        static unsigned _CalcBVHTotalSize(const BVHNode* root) {
            unsigned thisSize = root->TotalSize();
            if (root->isLeaf) return thisSize;

            if (root->lchild != nullptr) thisSize += _CalcBVHTotalSize(root->lchild);
            if (root->rchild != nullptr) thisSize += _CalcBVHTotalSize(root->rchild);

            return thisSize;
        }

        static void* _CopyBVH(const BVHNode* node, void* dst) {


            unsigned thisSize = node->TotalSize();

            char* dst_ = (char*)dst;
            dst_ += thisSize;

            if (node->isLeaf) {
                CUDA_CHECK(cudaMemcpy(dst, node, thisSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
                return dst_;
            }

            BVHNode n = *node;

            if (node->lchild != nullptr) {
                n.lchild = (BVHNode*)dst_;
                dst_ = (char*)_CopyBVH(node->lchild, dst_);
            }


            if (node->rchild != nullptr) {
                n.rchild = (BVHNode*)dst_;
                dst_ = (char*)_CopyBVH(node->rchild, dst_);
            }

            CUDA_CHECK(cudaMemcpy(dst, &n, thisSize, cudaMemcpyKind::cudaMemcpyHostToDevice));


            return dst_;
        }

    public:

        CUDATraceable(const Scene& scn) {
            //Allocate scene header
            auto si_payloadsize = scn.objects.size() * sizeof(void*);
            auto si_size = si_payloadsize + sizeof(SceneInfoHeader);
            _SceneInfo = std::make_unique<CUDABuffer>(si_size);

            //Fill header data
            auto siheader = (SceneInfoHeader*)_SceneInfo->cpu_data();
            siheader->skyLight = scn.skyLight;
            siheader->meshCnt = scn.objects.size();

            if (!scn.envMap.empty()) {
                //Assume rgb
                _EnvMap = std::make_unique<CUDABuffer>(scn.envMap.size() * sizeof(float));
                memcpy(_EnvMap->mutable_cpu_data(), scn.envMap.data(), _EnvMap->size());
                siheader->envMap = (float*)_EnvMap->gpu_data();
                siheader->envMapDim = scn.envMapDim;
            }
            else {
                siheader->envMap = nullptr;
            }

            auto siptable = (const void**)(siheader + 1);

            //Alloc mesh data and fill pointer tables
            for (unsigned i = 0; i < siheader->meshCnt; i++) {
                auto& mesh = scn.objects[i];
                auto verticesCnt = mesh.GetVertices().size();
                auto indicesCnt = mesh.GetIndices().size();

                auto meshSize = sizeof(MeshInfoHeader)
                    + sizeof(Vertex) * verticesCnt
                    + sizeof(std::uint32_t) * indicesCnt
                    ;
                auto totalSize = meshSize + _CalcBVHTotalSize(mesh.GetBVH());
                ;

                _MeshInfoArr.push_back(std::make_unique<CUDABuffer>(totalSize));
                auto cudabuffer = _MeshInfoArr.back().get();
                auto gpuBuffer = cudabuffer->gpu_data();
                auto buffer = cudabuffer->mutable_cpu_data();
                auto& mheader = *(MeshInfoHeader*)buffer;

                mheader.aabb = mesh.GetAABB();
                mheader.material = mesh.GetMat();
                mheader.transform = mesh.transform;
                mheader.vertexCnt = verticesCnt;
                mheader.indicesCnt = indicesCnt;

                auto vptr = mheader.GetVertices();
                memcpy(vptr, mesh.GetVertices().data(), verticesCnt * sizeof(Vertex));

                auto iptr = (std::uint32_t*)((char*)vptr + verticesCnt * sizeof(Vertex));
                memcpy(iptr, mesh.GetIndices().data(), indicesCnt * sizeof(std::uint32_t));

                //for (int i = 0; i < verticesCnt; i++) {
                //    auto& v = vptr[i];
                //
                //}

                mheader.bvh = (BVHNode*)(((char*)gpuBuffer) + meshSize);

                //Upload
                cudabuffer->gpu_data();

                auto res = _CopyBVH(mesh.GetBVH(), mheader.bvh);

                //Upload to gpu
                siptable[i] = gpuBuffer;
            }
        }


        virtual ~CUDATraceable() override {}

        const void* GetGPUData() const { return _SceneInfo->gpu_data(); }

        MeshInfoHeader& GetMeshMutable(unsigned idx) {
            auto ptr = _MeshInfoArr[idx]->mutable_cpu_data();
            return *as<MeshInfoHeader>(ptr);
        }

        const MeshInfoHeader& GetMesh(unsigned idx) {
            auto ptr = _MeshInfoArr[idx]->cpu_data();
            return *as<MeshInfoHeader>(ptr);
        }


        SceneInfoHeader& GetInfoMutable() {
            auto ptr = _SceneInfo->mutable_cpu_data();
            return *as<SceneInfoHeader>(ptr);
        }

        const SceneInfoHeader& GetInfo() {
            auto ptr = _SceneInfo->cpu_data();
            return *as<SceneInfoHeader>(ptr);
        }
    };

    
 

    class CUDATracerProg : public IPathTracer {
    public:
        CUDATracerProg() ;
        virtual ~CUDATracerProg() override {}


        virtual ITraceable* CreateTraceable(const Scene& scene) const override {
            return new CUDATraceable(scene);
        }
    
        virtual void Trace(
            //const TypedBuffer<Camera>& cam,
            const ITraceable& scn,
            TypedBuffer<PathTraceSettings>& settings,
            float* accBuffer,
            char* buffer
        ) override;
    };


} // namespace CUDATracer {

