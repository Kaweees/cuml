/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data.hpp"
#include "part_descriptor.hpp"

#include <cuml/common/export.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

namespace MLCommon {
namespace Matrix {
namespace opg {

CUML_EXPORT void gatherPart(const raft::handle_t& h,
                            float* gatheredPart,
                            std::vector<Matrix::Data<float>*>& parts,
                            Matrix::PartDescriptor& desc,
                            int partIndex,
                            int rootRank,
                            int myRank,
                            cudaStream_t stream);

CUML_EXPORT void allGatherPart(const raft::handle_t& h,
                               float* gatheredPart,
                               std::vector<Matrix::Data<float>*>& parts,
                               Matrix::PartDescriptor& desc,
                               int partIndex,
                               int myRank,
                               cudaStream_t stream);
CUML_EXPORT void gather(const raft::handle_t& h,
                        float* gatheredMatrix,
                        std::vector<Matrix::Data<float>*>& parts,
                        Matrix::PartDescriptor& desc,
                        int rootRank,
                        int myRank,
                        cudaStream_t stream);

CUML_EXPORT void allGather(const raft::handle_t& h,
                           float* gatheredMatrix,
                           std::vector<Matrix::Data<float>*>& parts,
                           Matrix::PartDescriptor& desc,
                           int myRank,
                           cudaStream_t stream);

CUML_EXPORT void allocate(const raft::handle_t& h,
                          std::vector<Matrix::Data<float>*>& parts,
                          Matrix::PartDescriptor& desc,
                          int myRank,
                          cudaStream_t stream);

CUML_EXPORT void deallocate(const raft::handle_t& h,
                            std::vector<Matrix::Data<float>*>& parts,
                            Matrix::PartDescriptor& desc,
                            int myRank,
                            cudaStream_t stream);

CUML_EXPORT void randomize(const raft::handle_t& h,
                           raft::random::Rng& r,
                           std::vector<Matrix::Data<float>*>& parts,
                           Matrix::PartDescriptor& desc,
                           int myRank,
                           cudaStream_t stream,
                           float low  = -1.0f,
                           float high = 1.0f);

CUML_EXPORT void reset(const raft::handle_t& h,
                       std::vector<Matrix::Data<float>*>& parts,
                       Matrix::PartDescriptor& desc,
                       int myRank,
                       cudaStream_t stream);

CUML_EXPORT void printRaw2D(
  float* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

CUML_EXPORT void print(const raft::handle_t& h,
                       std::vector<Matrix::Data<float>*>& parts,
                       Matrix::PartDescriptor& desc,
                       const char* matrixName,
                       int myRank,
                       cudaStream_t stream);

//------------------------------------------------------------------------------

CUML_EXPORT void gatherPart(const raft::handle_t& h,
                            double* gatheredPart,
                            std::vector<Matrix::Data<double>*>& parts,
                            Matrix::PartDescriptor& desc,
                            int partIndex,
                            int rootRank,
                            int myRank,
                            cudaStream_t stream);

CUML_EXPORT void allGatherPart(const raft::handle_t& h,
                               double* gatheredPart,
                               std::vector<Matrix::Data<double>*>& parts,
                               Matrix::PartDescriptor& desc,
                               int partIndex,
                               int myRank,
                               cudaStream_t stream);

CUML_EXPORT void gather(const raft::handle_t& h,
                        double* gatheredMatrix,
                        std::vector<Matrix::Data<double>*>& parts,
                        Matrix::PartDescriptor& desc,
                        int rootRank,
                        int myRank,
                        cudaStream_t stream);

CUML_EXPORT void allGather(const raft::handle_t& h,
                           double* gatheredMatrix,
                           std::vector<Matrix::Data<double>*>& parts,
                           Matrix::PartDescriptor& desc,
                           int myRank,
                           cudaStream_t stream);

CUML_EXPORT void allocate(const raft::handle_t& h,
                          std::vector<Matrix::Data<double>*>& parts,
                          Matrix::PartDescriptor& desc,
                          int myRank,
                          cudaStream_t stream);

CUML_EXPORT void deallocate(const raft::handle_t& h,
                            std::vector<Matrix::Data<double>*>& parts,
                            Matrix::PartDescriptor& desc,
                            int myRank,
                            cudaStream_t stream);

CUML_EXPORT void randomize(const raft::handle_t& h,
                           raft::random::Rng& r,
                           std::vector<Matrix::Data<double>*>& parts,
                           Matrix::PartDescriptor& desc,
                           int myRank,
                           cudaStream_t stream,
                           double low  = -1.0,
                           double high = 1.0);

CUML_EXPORT void reset(const raft::handle_t& h,
                       std::vector<Matrix::Data<double>*>& parts,
                       Matrix::PartDescriptor& desc,
                       int myRank,
                       cudaStream_t stream);

CUML_EXPORT void printRaw2D(
  double* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

CUML_EXPORT void print(const raft::handle_t& h,
                       std::vector<Matrix::Data<double>*>& parts,
                       Matrix::PartDescriptor& desc,
                       const char* matrixName,
                       int myRank,
                       cudaStream_t stream);
}  // end namespace opg
}  // namespace Matrix
}  // end namespace MLCommon
