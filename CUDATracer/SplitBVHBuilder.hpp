/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <vector>

#include "Types.hpp"

#define FW_F32_MIN          (1.175494351e-38f)
#define FW_F32_MAX          (3.402823466e+38f)
#define NULL 0
#define FW_ASSERT(X) ((void)0)  

namespace CUDATracer{
    using S32 = std::int32_t;
    using F32 = float;

    struct Stats
    {
        Stats() { clear(); }
        void clear() { memset(this, 0, sizeof(Stats)); }
        void print() const {} //printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n", branchingFactor, numLeafNodes + numInnerNodes, numLeafNodes, numInnerNodes, SAHCost, 1.f*numChildNodes / max1i(numInnerNodes, 1), 1.f*numTris / max1i(numLeafNodes, 1)); }

        F32     SAHCost;           // Surface Area Heuristic cost
        S32     branchingFactor;
        S32     numInnerNodes;
        S32     numLeafNodes;
        S32     numChildNodes;
        S32     numTris;
    };

    struct BuildParams
    {
        Stats* stats;
        bool        enablePrints;
        F32         splitAlpha;     // spatial split area threshold, see Nvidia paper on SBVH by Martin Stich, usually 0.05

        BuildParams(void)
        {
            stats = nullptr;
            enablePrints = true;
            splitAlpha = 1.0e-5f;
        }

    };

class SplitBVHBuilder
{
	
private:
	enum
	{
		MaxDepth = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins = 32,
	};

	struct Reference   /// a AABB bounding box enclosing 1 triangle, a reference can be duplicated by a split to be contained in 2 AABB boxes
	{
		S32                 triIdx;
		AABB                bounds;

		Reference(void) : triIdx(-1) {}  /// constructor
	};

	struct NodeSpec
	{
		S32                 numRef;   // number of references contained by node
		AABB                bounds;

		NodeSpec(void) : numRef(0) {}
	};

	struct ObjectSplit
	{
		F32                 sah;   // cost
		S32                 sortDim;  // axis along which triangles are sorted
		S32                 numLeft;  // number of triangles (references) in left child
		AABB                leftBounds;
		AABB                rightBounds;

		ObjectSplit(void) : sah(FW_F32_MAX), sortDim(0), numLeft(0) {}
	};

	struct SpatialSplit
	{
		F32                 sah;
		S32                 dim;   /// split axis
		F32                 pos;   /// position of split along axis (dim)

		SpatialSplit(void) : sah(FW_F32_MAX), dim(0), pos(0.0f) {}
	};

	struct SpatialBin
	{
		AABB                bounds;
		S32                 enter;
		S32                 exit;
	};

public:
	SplitBVHBuilder(Mesh& bvh, const BuildParams& params);
	~SplitBVHBuilder(void);

	BVHNode*                run();

private:
    static int              sortCompare(void* data, int idxA, int idxB);
	static void             sortSwap(void* data, int idxA, int idxB);

	BVHNode*                buildNode(const NodeSpec& spec, int level, F32 progressStart, F32 progressEnd);
	BVHNode*                createLeaf(const NodeSpec& spec);

	ObjectSplit             findObjectSplit(const NodeSpec& spec, F32 nodeSAH);
	void                    performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

	SpatialSplit            findSpatialSplit(const NodeSpec& spec, F32 nodeSAH);
	void                    performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
	void                    splitReference(Reference& left, Reference& right, const Reference& ref, int dim, F32 pos);

private:
	SplitBVHBuilder(const SplitBVHBuilder&); // forbidden
	SplitBVHBuilder&        operator=           (const SplitBVHBuilder&); // forbidden

private:
	Mesh&                    m_bvh;
	const Platform&         m_platform;
	const BuildParams& m_params;

	std::vector<Reference>      m_refStack;
	F32                     m_minOverlap;
	std::vector<AABB>           m_rightBounds;
	S32                     m_sortDim;
	SpatialBin              m_bins[3][NumSpatialBins];

	//Timer                   m_progressTimer;
	S32                     m_numDuplicates;
};


class BVHNode
{
public:
    BVHNode() : m_probability(1.f), m_parentProbability(1.f), m_treelet(-1), m_index(-1) {}
    virtual bool        isLeaf() const = 0;
    virtual S32         getNumChildNodes() const = 0;
    virtual BVHNode* getChildNode(S32 i) const = 0;
    virtual S32         getNumTriangles() const { return 0; }

    float       getArea() const { return m_bounds.area(); }

    AABB        m_bounds;

    // These are somewhat experimental, for some specific test and may be invalid...
    float       m_probability;          // probability of coming here (widebvh uses this)
    float       m_parentProbability;    // probability of coming to parent (widebvh uses this)

    int         m_treelet;              // for queuing tests (qmachine uses this)
    int         m_index;                // in linearized tree (qmachine uses this)

    // Subtree functions
    int     getSubtreeSize(BVH_STAT stat = BVH_STAT_NODE_COUNT) const;
    //void    computeSubtreeProbabilities(const Platform& p, float parentProbability, float& sah);
    //float   computeSubtreeSAHCost(const Platform& p) const;     // NOTE: assumes valid probabilities
    void    deleteSubtree();

    void    assignIndicesDepthFirst(S32 index = 0, bool includeLeafNodes = true);
    void    assignIndicesBreadthFirst(S32 index = 0, bool includeLeafNodes = true);
};


class InnerNode : public BVHNode
{
public:
    InnerNode(const AABB& bounds, BVHNode* child0, BVHNode* child1) { m_bounds = bounds; m_children[0] = child0; m_children[1] = child1; }

    bool        isLeaf() const { return false; }
    S32         getNumChildNodes() const { return 2; }
    BVHNode* getChildNode(S32 i) const { FW_ASSERT(i >= 0 && i < 2); return m_children[i]; }

    BVHNode* m_children[2];
};


class LeafNode : public BVHNode
{
public:
    LeafNode(const AABB& bounds, int lo, int hi) { m_bounds = bounds; m_lo = lo; m_hi = hi; }
    LeafNode(const LeafNode& s) { *this = s; }

    bool        isLeaf() const { return true; }
    S32         getNumChildNodes() const { return 0; }  // leafnode has 0 children
    BVHNode* getChildNode(S32) const { return NULL; }

    S32         getNumTriangles() const { return m_hi - m_lo; }
    S32         m_lo;  // lower index in triangle list
    S32         m_hi;  // higher index in triangle list
};

}
	