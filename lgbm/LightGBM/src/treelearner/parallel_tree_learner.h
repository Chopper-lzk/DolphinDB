/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_

#include <LightGBM/network.h>
#include <LightGBM/utils/array_args.h>

#include <cstring>
#include <memory>
#include <vector>

#include "cuda_tree_learner.h"
#include "gpu_tree_learner.h"
#include "serial_tree_learner.h"

namespace LightGBM {
class CostEfficientGradientBoosting;
/*!
* \brief Feature parallel learning algorithm.
*        Different machine will find best split on different features, then sync global best split
*        It is recommonded used when #data is small or #feature is large
*/
template <typename TREELEARNER_T>
class FeatureParallelTreeLearner: public TREELEARNER_T {
 public:
  explicit FeatureParallelTreeLearner(const Config* config);
  ~FeatureParallelTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;

 protected:
  void BeforeTrain() override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) override;

 private:
  /*! \brief rank of local machine */
  int rank_;
  /*! \brief Number of machines of this parallel task */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
};

/*!
* \brief Data parallel learning algorithm.
*        Workers use local data to construct histograms locally, then sync up global histograms.
*        It is recommonded used when #data is large or #feature is small
*/
template <typename TREELEARNER_T>
class DataParallelTreeLearner: public TREELEARNER_T {
 public:
  explicit DataParallelTreeLearner(const Config* config);
  ~DataParallelTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetConfig(const Config* config) override;

 protected:
  void BeforeTrain() override;
  void FindBestSplits(const Tree* tree) override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const override {
    if (leaf_idx >= 0) {
      return global_data_count_in_leaf_[leaf_idx];
    } else {
      return 0;
    }
  }

 private:
  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief Number of machines of this parallel task */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
  /*! \brief different machines will aggregate histograms for different features,
       use this to mark local aggregate features*/
  std::vector<bool> is_feature_aggregated_;
  /*! \brief Block start index for reduce scatter */
  std::vector<comm_size_t> block_start_;
  /*! \brief Block size for reduce scatter */
  std::vector<comm_size_t> block_len_;
  /*! \brief Write positions for feature histograms */
  std::vector<comm_size_t> buffer_write_start_pos_;
  /*! \brief Read positions for local feature histograms */
  std::vector<comm_size_t> buffer_read_start_pos_;
  /*! \brief Size for reduce scatter */
  comm_size_t reduce_scatter_size_;
  /*! \brief Store global number of data in leaves  */
  std::vector<data_size_t> global_data_count_in_leaf_;
};

/*!
* \brief Voting based data parallel learning algorithm.
* Like data parallel, but not aggregate histograms for all features.
* Here using voting to reduce features, and only aggregate histograms for selected features.
* When #data is large and #feature is large, you can use this to have better speed-up
*/
template <typename TREELEARNER_T>
class VotingParallelTreeLearner: public TREELEARNER_T {
 public:
  explicit VotingParallelTreeLearner(const Config* config);
  ~VotingParallelTreeLearner() { }
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetConfig(const Config* config) override;

 protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void FindBestSplits(const Tree* tree) override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const override {
    if (leaf_idx >= 0) {
      return global_data_count_in_leaf_[leaf_idx];
    } else {
      return 0;
    }
  }
  /*!
  * \brief Perform global voting
  * \param leaf_idx index of leaf
  * \param splits All splits from local voting
  * \param out Result of gobal voting, only store feature indices
  */
  void GlobalVoting(int leaf_idx, const std::vector<LightSplitInfo>& splits,
    std::vector<int>* out);
  /*!
  * \brief Copy local histgram to buffer
  * \param smaller_top_features Selected features for smaller leaf
  * \param larger_top_features Selected features for larger leaf
  */
  void CopyLocalHistogram(const std::vector<int>& smaller_top_features,
    const std::vector<int>& larger_top_features);

 private:
  /*! \brief Tree config used in local mode */
  Config local_config_;
  /*! \brief Voting size */
  int top_k_;
  /*! \brief Rank of local machine*/
  int rank_;
  /*! \brief Number of machines */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
  /*! \brief different machines will aggregate histograms for different features,
  use this to mark local aggregate features*/
  std::vector<bool> smaller_is_feature_aggregated_;
  /*! \brief different machines will aggregate histograms for different features,
  use this to mark local aggregate features*/
  std::vector<bool> larger_is_feature_aggregated_;
  /*! \brief Block start index for reduce scatter */
  std::vector<comm_size_t> block_start_;
  /*! \brief Block size for reduce scatter */
  std::vector<comm_size_t> block_len_;
  /*! \brief Read positions for feature histgrams at smaller leaf */
  std::vector<comm_size_t> smaller_buffer_read_start_pos_;
  /*! \brief Read positions for feature histgrams at larger leaf */
  std::vector<comm_size_t> larger_buffer_read_start_pos_;
  /*! \brief Size for reduce scatter */
  comm_size_t reduce_scatter_size_;
  /*! \brief Store global number of data in leaves  */
  std::vector<data_size_t> global_data_count_in_leaf_;
  /*! \brief Store global split information for smaller leaf  */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_global_;
  /*! \brief Store global split information for larger leaf  */
  std::unique_ptr<LeafSplits> larger_leaf_splits_global_;
  /*! \brief Store global histogram for smaller leaf  */
  std::unique_ptr<FeatureHistogram[]> smaller_leaf_histogram_array_global_;
  /*! \brief Store global histogram for larger leaf  */
  std::unique_ptr<FeatureHistogram[]> larger_leaf_histogram_array_global_;

  std::vector<hist_t> smaller_leaf_histogram_data_;
  std::vector<hist_t> larger_leaf_histogram_data_;
  std::vector<FeatureMetainfo> feature_metas_;
};

// To-do: reduce the communication cost by using bitset to communicate.
inline void SyncUpGlobalBestSplit(char* input_buffer_, char* output_buffer_, SplitInfo* smaller_best_split, SplitInfo* larger_best_split, int max_cat_threshold) {
  // sync global best info
  int size = SplitInfo::Size(max_cat_threshold);
  smaller_best_split->CopyTo(input_buffer_);
  larger_best_split->CopyTo(input_buffer_ + size);
  Network::Allreduce(input_buffer_, size * 2, size, output_buffer_,
                     [] (const char* src, char* dst, int size, comm_size_t len) {
    comm_size_t used_size = 0;
    LightSplitInfo p1, p2;
    while (used_size < len) {
      p1.CopyFrom(src);
      p2.CopyFrom(dst);
      if (p1 > p2) {
        std::memcpy(dst, src, size);
      }
      src += size;
      dst += size;
      used_size += size;
    }
  });
  // copy back
  smaller_best_split->CopyFrom(output_buffer_);
  larger_best_split->CopyFrom(output_buffer_ + size);
}

class DBParallelTreeLearner: public SerialTreeLearner{
 public:
  friend CostEfficientGradientBoosting;

  DBParallelTreeLearner();
  explicit DBParallelTreeLearner(const Config* config);
  ~DBParallelTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t *hessians) override;
  TrainStatus trainStatus = NoneTrain;
  int train_leaf = 0;
  void setSplitInfo(data_size_t num_data_in_leaf, double sum_gradients, double sum_hessians);
  std::vector<FeatureHistogram*> smaller_leaf_histogram_array_reduce;
  std::unique_ptr<Tree> tempTree;
  int left_leaf = 0;
  int cur_depth = 1;
  int right_leaf = -1;
  void getSplitInfo(data_size_t* num_data_in_leaf, double* sum_gradients, double* sum_hessians){
    *num_data_in_leaf = smaller_leaf_splits_->num_data_in_leaf();
    *sum_gradients = smaller_leaf_splits_->sum_gradients();
    *sum_hessians = smaller_leaf_splits_->sum_hessians();
  }
  SplitInfo local_smaller_best_split, local_larger_best_split;
 protected:
  void BeforeTrain();
  void FindBestSplits(const Tree* tree);
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree);
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf);

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const {
    if (leaf_idx >= 0) {
      return global_data_count_in_leaf_[leaf_idx];
    } else {
      return 0;
    }
  }
  inline void SetForcedSplit(const Json* forced_split_json) {
    if (forced_split_json != nullptr && !forced_split_json->is_null()) {
      forced_split_json_ = forced_split_json;
    } else {
      forced_split_json_ = nullptr;
    }
  }


  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    if (subset == nullptr) {
      data_partition_->SetUsedDataIndices(used_indices, num_data);
      share_state_->SetUseSubrow(false);
    } else {
      ResetTrainingDataInner(subset, share_state_->is_constant_hessian, false);
      share_state_->SetUseSubrow(true);
      share_state_->SetSubrowCopied(false);
      share_state_->bagging_use_indices = used_indices;
      share_state_->bagging_indices_cnt = num_data;
    }
  }

  void ResetTrainingDataInner(const Dataset* train_data,
                                               bool is_constant_hessian,
                                               bool reset_multi_val_bin);
  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    if (tree->num_leaves() <= 1) {
      return;
    }
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        out_score[tmp_idx[j]] += output;
      }
    }
  }

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

  /*! \brief Get output of parent node, used for path smoothing */
  double GetParentOutput(const Tree* tree, const LeafSplits* leaf_splits) const;

 protected:
  void ComputeBestSplitForFeature(FeatureHistogram* histogram_array_,
                                  int feature_index, int real_fidx,
                                  int8_t is_feature_used, int num_data,
                                  const LeafSplits* leaf_splits,
                                  SplitInfo* best_split, double parent_output);


  void GetShareStates(const Dataset* dataset, bool is_constant_hessian, bool is_first_time);

  void RecomputeBestSplitForLeaf(int leaf, SplitInfo* split);

  virtual bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf);

  virtual void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract);

  void SplitInner(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf,
                  bool update_cnt);

  /* Force splits with forced_split_json dict and then return num splits forced.*/
  int32_t ForceSplits(Tree* tree, int* left_leaf, int* right_leaf,
                      int* cur_depth);
  /*! \brief number of data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  /*! \brief training data */
  const Dataset* train_data_;
  /*! \brief gradients of current iteration */
  const score_t* gradients_;
  /*! \brief hessians of current iteration */
  const score_t* hessians_;
  /*! \brief training data partition on leaves */
  std::unique_ptr<DataPartition> data_partition_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;

  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;
  /*! \brief store best split per feature for all leaves */
  std::vector<SplitInfo> splits_per_leaf_;
  /*! \brief stores minimum and maximum constraints for each leaf */
  std::unique_ptr<LeafConstraintsBase> constraints_;

  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplits> larger_leaf_splits_;

  /*! \brief gradients of current iteration, ordered for cache optimized */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> ordered_hessians_;

  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const Config* config_;
  ColSampler col_sampler_;
  const Json* forced_split_json_;
  std::unique_ptr<TrainingShareStates> share_state_;
  std::unique_ptr<CostEfficientGradientBoosting> cegb_;

  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief Number of machines of this parallel task */
  int num_machines_;
  /*! \brief Buffer for network send */
  // std::vector<char> input_buffer_;
  // /*! \brief Buffer for network receive */
  // std::vector<char> output_buffer_;
  // /*! \brief different machines will aggregate histograms for different features,
  //      use this to mark local aggregate features*/
  // std::vector<bool> is_feature_aggregated_;
  // /*! \brief Block start index for reduce scatter */
  // std::vector<comm_size_t> block_start_;
  // /*! \brief Block size for reduce scatter */
  // std::vector<comm_size_t> block_len_;
  // /*! \brief Write positions for feature histograms */
  // std::vector<comm_size_t> buffer_write_start_pos_;
  // /*! \brief Read positions for local feature histograms */
  // std::vector<comm_size_t> buffer_read_start_pos_;
  // /*! \brief Size for reduce scatter */
  // comm_size_t reduce_scatter_size_;
  // /*! \brief Store global number of data in leaves  */
  std::vector<data_size_t> global_data_count_in_leaf_;
};


}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_
