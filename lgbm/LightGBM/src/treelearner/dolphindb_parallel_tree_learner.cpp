/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>

#include "parallel_tree_learner.h"

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>

#include "cost_effective_gradient_boosting.hpp"
namespace LightGBM {


DBParallelTreeLearner::DBParallelTreeLearner(const Config* config)
  :SerialTreeLearner(config), config_(config), col_sampler_(config){

}
DBParallelTreeLearner::~DBParallelTreeLearner() {
}


void DBParallelTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  int max_cache_size = 0;
  // Get the max size of pool
  if (config_->histogram_pool_size <= 0) {
    max_cache_size = config_->num_leaves;
  } else {
    size_t total_histogram_size = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      total_histogram_size += kHistEntrySize * train_data_->FeatureNumBin(i);
    }
    max_cache_size = static_cast<int>(config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
  }
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, config_->num_leaves);

  // push split information for all leaves
  best_split_per_leaf_.resize(config_->num_leaves);
  constraints_.reset(LeafConstraintsBase::Create(config_, config_->num_leaves, train_data_->num_features()));

  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_data(), config_));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_data(), config_));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, config_->num_leaves));
  col_sampler_.SetTrainingData(train_data_);
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);

  GetShareStates(train_data_, is_constant_hessian, true);
  histogram_pool_.DynamicChangeSize(train_data_,
  share_state_->num_hist_total_bin(),
  share_state_->feature_hist_offsets(),
  config_, max_cache_size, config_->num_leaves);
  Log::Info("Number of data points in the train set: %d, number of used features: %d", num_data_, num_features_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }
  // Get local rank and global machine size
  //rank_ = Network::rank();
  //num_machines_ = Network::num_machines();

  // need to be able to hold smaller and larger best splits in SyncUpGlobalBestSplit
  // allocate buffer for communication
  // size_t buffer_size = std::max(histogram_size, split_info_size);

  // input_buffer_.resize(buffer_size);
  // output_buffer_.resize(buffer_size);

  // is_feature_aggregated_.resize(this->num_features_);

  // //block_start_.resize(num_machines_);
  // //block_len_.resize(num_machines_);

  // buffer_write_start_pos_.resize(this->num_features_);
  // buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

void DBParallelTreeLearner::BeforeTrain() {
  Common::FunctionTimer fun_timer("SerialTreeLearner::BeforeTrain", global_timer);
  // reset histogram pool
  histogram_pool_.ResetMap();

  col_sampler_.ResetByTree();
  train_data_->InitTrain(col_sampler_.is_feature_used_bytree(), share_state_.get());
  // initialize data partition
  data_partition_->Init();

  constraints_->Reset();

  // reset the splits for leaves
  for (int i = 0; i < config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);

  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
  }

  larger_leaf_splits_->Init();

  /*...................*/

  // generate feature partition for current tree
  // std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  // std::vector<int> num_bins_distributed(num_machines_, 0);
  // for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
  //   int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
  //   if (inner_feature_index == -1) { continue; }
  //   if (this->col_sampler_.is_feature_used_bytree()[inner_feature_index]) {
  //     int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
  //     feature_distribution[cur_min_machine].push_back(inner_feature_index);
  //     auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
  //     if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetMostFreqBin() == 0) {
  //       num_bin -= 1;
  //     }
  //     num_bins_distributed[cur_min_machine] += num_bin;
  //   }
  //   is_feature_aggregated_[inner_feature_index] = false;
  // }
  // // get local used feature
  // for (auto fid : feature_distribution[rank_]) {
  //   is_feature_aggregated_[fid] = true;
  // }

  // get block start and block len for reduce scatter
  // reduce_scatter_size_ = 0;
  // for (int i = 0; i < num_machines_; ++i) {
  //   block_len_[i] = 0;
  //   for (auto fid : feature_distribution[i]) {
  //     auto num_bin = this->train_data_->FeatureNumBin(fid);
  //     if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
  //       num_bin -= 1;
  //     }
  //     block_len_[i] += num_bin * kHistEntrySize;
  //   }
  //   reduce_scatter_size_ += block_len_[i];
  // }

  // block_start_[0] = 0;
  // for (int i = 1; i < num_machines_; ++i) {
  //   block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  // }

  // // get buffer_write_start_pos_
  // int bin_size = 0;
  // for (int i = 0; i < num_machines_; ++i) {
  //   for (auto fid : feature_distribution[i]) {
  //     buffer_write_start_pos_[fid] = bin_size;
  //     auto num_bin = this->train_data_->FeatureNumBin(fid);
  //     if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
  //       num_bin -= 1;
  //     }
  //     bin_size += num_bin * kHistEntrySize;
  //   }
  // }

  // // get buffer_read_start_pos_
  // bin_size = 0;
  // for (auto fid : feature_distribution[rank_]) {
  //   buffer_read_start_pos_[fid] = bin_size;
  //   auto num_bin = this->train_data_->FeatureNumBin(fid);
  //   if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
  //     num_bin -= 1;
  //   }
  //   bin_size += num_bin * kHistEntrySize;
  // }

  // ddb
  trainStatus = sumSplitInfo;
  return;
  // sync global data sumup info
  // std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
  //                                              this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  // int size = sizeof(data);
  // std::memcpy(input_buffer_.data(), &data, size);

  // // global sumup reduce
  // Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
  //   comm_size_t used_size = 0;
  //   const std::tuple<data_size_t, double, double> *p1;
  //   std::tuple<data_size_t, double, double> *p2;
  //   while (used_size < len) {
  //     p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
  //     p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
  //     std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
  //     std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
  //     std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
  //     src += type_size;
  //     dst += type_size;
  //     used_size += type_size;
  //   }
  // });
  // copy back
  // std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);
  // // set global sumup info
  // this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // // init global data count in leaf
  // global_data_count_in_leaf_[0] = std::get<0>(data);
}

void DBParallelTreeLearner::setSplitInfo(data_size_t num_data_in_leaf, double sum_gradients, double sum_hessians){
  this->smaller_leaf_splits_->Init(sum_gradients, sum_hessians);
  global_data_count_in_leaf_[0] = num_data_in_leaf;
}

Tree* DBParallelTreeLearner::Train(const score_t* gradients, const score_t *hessians) {
  if(trainStatus == NoneTrain){
    Common::FunctionTimer fun_timer("SerialTreeLearner::Train", global_timer);
    gradients_ = gradients;
    hessians_ = hessians;
    int num_threads = OMP_NUM_THREADS();
    if (share_state_->num_threads != num_threads && share_state_->num_threads > 0) {
      Log::Warning(
          "Detected that num_threads changed during training (from %d to %d), "
          "it may cause unexpected errors.",
          share_state_->num_threads, num_threads);
    }
    share_state_->num_threads = num_threads;
    train_leaf = 0;
    // root leaf
    left_leaf = 0;
    cur_depth = 1;
    // only root leaf can be splitted on first time
    right_leaf = -1;
    // some initial works before training
    BeforeTrain();
    
    return nullptr;
    
  }
  if(trainStatus == sumSplitInfo){
    bool track_branch_features = !(config_->interaction_constraints_vector.empty());
    tempTree =  std::unique_ptr<Tree>(new Tree(config_->num_leaves, track_branch_features));
    auto tree_ptr = tempTree.get();
    constraints_->ShareTreePointer(tree_ptr);

  // root leaf
    left_leaf = 0;
    cur_depth = 1;
    // only root leaf can be splitted on first time
    right_leaf = -1;


    for (; train_leaf < config_->num_leaves - 1; ++train_leaf) {
    // some initial works before finding best split
      if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
        // find best threshold for every feature
        FindBestSplits(tree_ptr);
        return nullptr;
      }
      // Get a leaf with max split gain
      int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
      // Get split information for best leaf
      const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
      // cannot split, quit
      if (best_leaf_SplitInfo.gain <= 0.0) {
        Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
        break;
      }
      // split tree with best leaf
      Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
      cur_depth = std::max(cur_depth, tempTree->leaf_depth(left_leaf));
    }
    Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tempTree->num_leaves(), cur_depth);
    trainStatus = NoneTrain;
    return tempTree.release();
  }
  if(trainStatus == sumHistograms){
    FindBestSplitsFromHistograms(this->col_sampler_.is_feature_used_bytree(), true, tempTree.get());

    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
        Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tempTree->num_leaves(), cur_depth);
      trainStatus = NoneTrain;

      return tempTree.release();
    }
    // split tree with best leaf
    Split(tempTree.get(), best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tempTree->leaf_depth(left_leaf));
    train_leaf ++;
    auto tree_ptr = tempTree.get();
    for (; train_leaf < config_->num_leaves - 1; ++train_leaf) {
      if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
        // find best threshold for every feature
        FindBestSplits(tree_ptr);
        return nullptr;
      }
      // Get a leaf with max split gain
      int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
      // Get split information for best leaf
      const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
      // cannot split, quit
      if (best_leaf_SplitInfo.gain <= 0.0) {
        Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
        Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tempTree->num_leaves(), cur_depth);
            trainStatus = NoneTrain;

        return tempTree.release();
      }
      // split tree with best leaf
      Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
      cur_depth = std::max(cur_depth, tempTree->leaf_depth(left_leaf));
    }
    Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tempTree->num_leaves(), cur_depth);
        trainStatus = NoneTrain;

    return tempTree.release();
  }
  
}

void DBParallelTreeLearner::FindBestSplits(const Tree*) {
  ConstructHistograms(
      this->col_sampler_.is_feature_used_bytree(), true);
  smaller_leaf_histogram_array_reduce.resize(num_features_, nullptr);
  // construct local histograms
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if (this->col_sampler_.is_feature_used_bytree()[feature_index] == false)
      continue;
    // copy to buffer
    smaller_leaf_histogram_array_reduce[this->train_data_->RealFeatureIndex(feature_index)] = &this->smaller_leaf_histogram_array_[feature_index];
    // std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
    //             this->smaller_leaf_histogram_array_[feature_index].RawData(),
    //             this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
  }
  trainStatus = sumHistograms;
  // Reduce scatter for histogram
  // Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(hist_t), block_start_.data(),
  //                        block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramSumReducer);
  // this->FindBestSplitsFromHistograms(
  //     this->col_sampler_.is_feature_used_bytree(), true, tree);
}

void DBParallelTreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool, const Tree* tree) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->share_state_->num_threads);
  std::vector<SplitInfo> larger_bests_per_thread(this->share_state_->num_threads);
  std::vector<int8_t> smaller_node_used_features =
      this->col_sampler_.GetByNode(tree, this->smaller_leaf_splits_->leaf_index());
  std::vector<int8_t> larger_node_used_features =
      this->col_sampler_.GetByNode(tree, this->larger_leaf_splits_->leaf_index());
  double smaller_leaf_parent_output = this->GetParentOutput(tree, this->smaller_leaf_splits_.get());
  double larger_leaf_parent_output = this->GetParentOutput(tree, this->larger_leaf_splits_.get());
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    //if (!is_feature_aggregated_[feature_index]) continue;
    if (!is_feature_used[feature_index]) {
      continue;
    }
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    // this->smaller_leaf_histogram_array_[feature_index].FromMemory(
    //   output_buffer_.data() + buffer_read_start_pos_[feature_index]);
    this->smaller_leaf_histogram_array_[feature_index].FromMemory((char*)smaller_leaf_histogram_array_reduce[real_feature_index]->RawData());

    this->train_data_->FixHistogram(feature_index,
                                    this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                    this->smaller_leaf_histogram_array_[feature_index].RawData());

    this->ComputeBestSplitForFeature(
        this->smaller_leaf_histogram_array_, feature_index, real_feature_index,
        smaller_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->leaf_index()),
        this->smaller_leaf_splits_.get(),
        &smaller_bests_per_thread[tid],
        smaller_leaf_parent_output);

    // only root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->leaf_index() < 0) continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    this->larger_leaf_histogram_array_[feature_index].Subtract(
      this->smaller_leaf_histogram_array_[feature_index]);

    this->ComputeBestSplitForFeature(
        this->larger_leaf_histogram_array_, feature_index, real_feature_index,
        larger_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->larger_leaf_splits_->leaf_index()),
        this->larger_leaf_splits_.get(),
        &larger_bests_per_thread[tid],
        larger_leaf_parent_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->leaf_index();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr &&  this->larger_leaf_splits_->leaf_index() >= 0) {
    leaf = this->larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }
  // SplitInfo smaller_best_split, larger_best_split;
  // smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()];
  // // find local best split for larger leaf
  // if (this->larger_leaf_splits_->leaf_index() >= 0) {
  //   larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()];
  // }
  // // trainStatus = syncUpBestSplit;
  // // local_smaller_best_split = smaller_best_split;
  // // local_larger_best_split = larger_best_split;
  // //return;
  // // sync global best info
  // //SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // set best split
  // this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()] = smaller_best_split;
  // if (this->larger_leaf_splits_->leaf_index() >= 0) {
  //   this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()] = larger_best_split;
  // }
}

void DBParallelTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  SplitInner(tree, best_Leaf, left_leaf, right_leaf, false);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

void DBParallelTreeLearner::GetShareStates(const Dataset* dataset,
                                       bool is_constant_hessian,
                                       bool is_first_time) {
  if (is_first_time) {
    share_state_.reset(dataset->GetShareStates(
        ordered_gradients_.data(), ordered_hessians_.data(),
        col_sampler_.is_feature_used_bytree(), is_constant_hessian,
        config_->force_col_wise, config_->force_row_wise));
  } else {
    CHECK_NOTNULL(share_state_);
    // cannot change is_hist_col_wise during training
    share_state_.reset(dataset->GetShareStates(
        ordered_gradients_.data(), ordered_hessians_.data(), col_sampler_.is_feature_used_bytree(),
        is_constant_hessian, share_state_->is_col_wise,
        !share_state_->is_col_wise));
  }
  CHECK_NOTNULL(share_state_);
}

bool DBParallelTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::BeforeFindBestSplit", global_timer);
  // check depth of current leaf
  if (config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // no enough data to continue
  if (num_data_in_right_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)
      && num_data_in_left_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
  parent_leaf_histogram_array_ = nullptr;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    // put parent(left) leaf's histograms into larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    // put parent(left) leaf's histograms to larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }
  return true;
}

void DBParallelTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::ConstructHistograms",
                                  global_timer);
  // construct smaller leaf
  hist_t* ptr_smaller_leaf_hist_data =
      smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  train_data_->ConstructHistograms(
      is_feature_used, smaller_leaf_splits_->data_indices(),
      smaller_leaf_splits_->num_data_in_leaf(), gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(), share_state_.get(),
      ptr_smaller_leaf_hist_data);
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    hist_t* ptr_larger_leaf_hist_data =
        larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    train_data_->ConstructHistograms(
        is_feature_used, larger_leaf_splits_->data_indices(),
        larger_leaf_splits_->num_data_in_leaf(), gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(), share_state_.get(),
        ptr_larger_leaf_hist_data);
  }
}

void DBParallelTreeLearner::SplitInner(Tree* tree, int best_leaf, int* left_leaf,
                                   int* right_leaf, bool update_cnt) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::SplitInner", global_timer);
  SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  const int inner_feature_index =
      train_data_->InnerFeatureIndex(best_split_info.feature);
  if (cegb_ != nullptr) {
    cegb_->UpdateLeafBestSplits(tree, best_leaf, &best_split_info,
                                &best_split_per_leaf_);
  }
  *left_leaf = best_leaf;
  auto next_leaf_id = tree->NextLeafId();

  // update before tree split
  constraints_->BeforeSplit(best_leaf, next_leaf_id,
                            best_split_info.monotone_type);

  bool is_numerical_split =
      train_data_->FeatureBinMapper(inner_feature_index)->bin_type() ==
      BinType::NumericalBin;
  if (is_numerical_split) {
    auto threshold_double = train_data_->RealThreshold(
        inner_feature_index, best_split_info.threshold);
    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           &best_split_info.threshold, 1,
                           best_split_info.default_left, next_leaf_id);
    if (update_cnt) {
      // don't need to update this in data-based parallel model
      best_split_info.left_count = data_partition_->leaf_count(*left_leaf);
      best_split_info.right_count = data_partition_->leaf_count(next_leaf_id);
    }
    // split tree, will return right leaf
    *right_leaf = tree->Split(
        best_leaf, inner_feature_index, best_split_info.feature,
        best_split_info.threshold, threshold_double,
        static_cast<double>(best_split_info.left_output),
        static_cast<double>(best_split_info.right_output),
        static_cast<data_size_t>(best_split_info.left_count),
        static_cast<data_size_t>(best_split_info.right_count),
        static_cast<double>(best_split_info.left_sum_hessian),
        static_cast<double>(best_split_info.right_sum_hessian),
        // store the true split gain in tree model
        static_cast<float>(best_split_info.gain + config_->min_gain_to_split),
        train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
        best_split_info.default_left);
  } else {
    std::vector<uint32_t> cat_bitset_inner =
        Common::ConstructBitset(best_split_info.cat_threshold.data(),
                                best_split_info.num_cat_threshold);
    std::vector<int> threshold_int(best_split_info.num_cat_threshold);
    for (int i = 0; i < best_split_info.num_cat_threshold; ++i) {
      threshold_int[i] = static_cast<int>(train_data_->RealThreshold(
          inner_feature_index, best_split_info.cat_threshold[i]));
    }
    std::vector<uint32_t> cat_bitset = Common::ConstructBitset(
        threshold_int.data(), best_split_info.num_cat_threshold);

    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           cat_bitset_inner.data(),
                           static_cast<int>(cat_bitset_inner.size()),
                           best_split_info.default_left, next_leaf_id);

    if (update_cnt) {
      // don't need to update this in data-based parallel model
      best_split_info.left_count = data_partition_->leaf_count(*left_leaf);
      best_split_info.right_count = data_partition_->leaf_count(next_leaf_id);
    }

    *right_leaf = tree->SplitCategorical(
        best_leaf, inner_feature_index, best_split_info.feature,
        cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
        cat_bitset.data(), static_cast<int>(cat_bitset.size()),
        static_cast<double>(best_split_info.left_output),
        static_cast<double>(best_split_info.right_output),
        static_cast<data_size_t>(best_split_info.left_count),
        static_cast<data_size_t>(best_split_info.right_count),
        static_cast<double>(best_split_info.left_sum_hessian),
        static_cast<double>(best_split_info.right_sum_hessian),
        // store the true split gain in tree model
        static_cast<float>(best_split_info.gain + config_->min_gain_to_split),
        train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
  }

#ifdef DEBUG
  CHECK(*right_leaf == next_leaf_id);
#endif

  // init the leaves that used on next iteration
  if (best_split_info.left_count < best_split_info.right_count) {
    CHECK_GT(best_split_info.left_count, 0);
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian,
                               best_split_info.left_output);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                              best_split_info.right_sum_gradient,
                              best_split_info.right_sum_hessian,
                              best_split_info.right_output);
  } else {
    CHECK_GT(best_split_info.right_count, 0);
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                               best_split_info.right_sum_gradient,
                               best_split_info.right_sum_hessian,
                               best_split_info.right_output);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                              best_split_info.left_sum_gradient,
                              best_split_info.left_sum_hessian,
                              best_split_info.left_output);
  }
  auto leaves_need_update = constraints_->Update(
      is_numerical_split, *left_leaf, *right_leaf,
      best_split_info.monotone_type, best_split_info.right_output,
      best_split_info.left_output, inner_feature_index, best_split_info,
      best_split_per_leaf_);
  // update leave outputs if needed
  for (auto leaf : leaves_need_update) {
    RecomputeBestSplitForLeaf(leaf, &best_split_per_leaf_[leaf]);
  }
}

void DBParallelTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                        data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK_EQ(bag_cnt, num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    int num_machines = Network::num_machines();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto index_mapper = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      if (cnt_leaf_data > 0) {
        // bag_mapper[index_mapper[i]]
        const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper, bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
        CHECK_GT(num_machines, 1);
        tree->SetLeafOutput(i, 0.0);
        n_nozeroworker_perleaf[i] = 0;
      }
    }
    if (num_machines > 1) {
      std::vector<double> outputs(tree->num_leaves());
      for (int i = 0; i < tree->num_leaves(); ++i) {
        outputs[i] = static_cast<double>(tree->LeafOutput(i));
      }
      outputs = Network::GlobalSum(&outputs);
      n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
      for (int i = 0; i < tree->num_leaves(); ++i) {
        tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
      }
    }
  }
}

void DBParallelTreeLearner::ComputeBestSplitForFeature(
    FeatureHistogram* histogram_array_, int feature_index, int real_fidx,
    int8_t is_feature_used, int num_data, const LeafSplits* leaf_splits,
    SplitInfo* best_split, double parent_output) {
  bool is_feature_numerical = train_data_->FeatureBinMapper(feature_index)
                                  ->bin_type() == BinType::NumericalBin;
  if (is_feature_numerical & !config_->monotone_constraints.empty()) {
    constraints_->RecomputeConstraintsIfNeeded(
        constraints_.get(), feature_index, ~(leaf_splits->leaf_index()),
        train_data_->FeatureNumBin(feature_index));
  }
  SplitInfo new_split;
  histogram_array_[feature_index].FindBestThreshold(
      leaf_splits->sum_gradients(), leaf_splits->sum_hessians(), num_data,
      constraints_->GetFeatureConstraint(leaf_splits->leaf_index(), feature_index), parent_output, &new_split);
  new_split.feature = real_fidx;
  if (cegb_ != nullptr) {
    new_split.gain -=
        cegb_->DetlaGain(feature_index, real_fidx, leaf_splits->leaf_index(),
                         num_data, new_split);
  }
  if (new_split.monotone_type != 0) {
    double penalty = constraints_->ComputeMonotoneSplitGainPenalty(
        leaf_splits->leaf_index(), config_->monotone_penalty);
    new_split.gain *= penalty;
  }
  // it is needed to filter the features after the above code.
  // Otherwise, the `is_splittable` in `FeatureHistogram` will be wrong, and cause some features being accidentally filtered in the later nodes.
  if (new_split > *best_split && is_feature_used) {
    *best_split = new_split;
  }
}

double DBParallelTreeLearner::GetParentOutput(const Tree* tree, const LeafSplits* leaf_splits) const {
  double parent_output;
  if (tree->num_leaves() == 1) {
    // for root leaf the "parent" output is its own output because we don't apply any smoothing to the root
    parent_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
      leaf_splits->sum_gradients(), leaf_splits->sum_hessians(), config_->lambda_l1,
      config_->lambda_l2, config_->max_delta_step, BasicConstraint(),
      config_->path_smooth, static_cast<data_size_t>(leaf_splits->num_data_in_leaf()), 0);
  } else {
    parent_output = leaf_splits->weight();
  }
  return parent_output;
}

void DBParallelTreeLearner::RecomputeBestSplitForLeaf(int leaf, SplitInfo* split) {
  FeatureHistogram* histogram_array_;
  if (!histogram_pool_.Get(leaf, &histogram_array_)) {
    Log::Warning(
        "Get historical Histogram for leaf %d failed, will skip the "
        "``RecomputeBestSplitForLeaf``",
        leaf);
    return;
  }
  double sum_gradients = split->left_sum_gradient + split->right_sum_gradient;
  double sum_hessians = split->left_sum_hessian + split->right_sum_hessian;
  int num_data = split->left_count + split->right_count;

  std::vector<SplitInfo> bests(share_state_->num_threads);
  LeafSplits leaf_splits(num_data, config_);
  leaf_splits.Init(leaf, sum_gradients, sum_hessians);

  // can't use GetParentOutput because leaf_splits doesn't have weight property set
  double parent_output = 0;
  if (config_->path_smooth > kEpsilon) {
    parent_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
      sum_gradients, sum_hessians, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
      BasicConstraint(), config_->path_smooth, static_cast<data_size_t>(num_data), 0);
  }

  OMP_INIT_EX();
// find splits
#pragma omp parallel for schedule(static) num_threads(share_state_->num_threads)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!col_sampler_.is_feature_used_bytree()[feature_index] ||
        !histogram_array_[feature_index].is_splittable()) {
      continue;
    }
    const int tid = omp_get_thread_num();
    int real_fidx = train_data_->RealFeatureIndex(feature_index);
    ComputeBestSplitForFeature(histogram_array_, feature_index, real_fidx, true,
                               num_data, &leaf_splits, &bests[tid], parent_output);

    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  auto best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  *split = bests[best_idx];
}

void DBParallelTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                              bool is_constant_hessian,
                                              bool reset_multi_val_bin) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK_EQ(num_features_, train_data_->num_features());

  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);
  if (reset_multi_val_bin) {
    col_sampler_.SetTrainingData(train_data_);
    GetShareStates(train_data_, is_constant_hessian, false);
  }

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  if (cegb_ != nullptr) {
    cegb_->Init();
  }
}

}  // namespace LightGBM
