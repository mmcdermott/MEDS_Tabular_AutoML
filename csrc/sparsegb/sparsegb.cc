// sparsegb: histogram gradient boosting (binary log-loss) for wide, sparse, mostly-zero tabular
// data.
//
// Same algorithm as LightGBM in MEDS-Tab's configuration: Newton boosting from the log-odds of the
// base rate, per-feature value histograms (<= max_bin bins, zero has its own bin), leaf-wise growth
// to num_leaves leaves, min_child_samples / min_sum_hessian / lambda_l2 split constraints,
// learning-rate shrinkage, per-tree bagging, early stopping on tuning log-loss. Columns can be
// grouped into slots (one code and aggregation, all windows); each tree then uses one column per
// slot (from precomputed layouts, or a sample redrawn every `refresh` trees).
//
// Engineering for ~1e5 columns and ~1e3 non-zeros per row:
//  * absent entries (zero bin) are never stored or scanned: a node's zero-bin statistics = node
//  totals - other bins;
//  * active features: a feature with fewer than min_child_samples non-zero rows in a node cannot
//  split it or any
//    descendant, so all its per-bin work is skipped there (exact);
//  * a row-major subset of the sampled columns, cut into feature chunks (16-bit bin ids local to
//    each chunk) with per-row chunk offsets, so each split is one parallel pass over chunks:
//    smaller child's histogram, larger child by subtraction from the parent, best split of both
//    children;
//  * root histograms by streaming rows; partitions through a per-row bin array of the split
//    feature, kept for later splits on it; software prefetching.
//
// C ABI (ctypes, MEDS_tabular_automl/sparsegb): sgb_train / sgb_predict / sgb_save / sgb_load /
// sgb_free.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

struct Feat {
  int nbin = 0;           // number of bins (<= max_bin)
  int zero_bin = 0;       // bin holding 0 and absent entries
  std::vector<float> ub;  // ub[k]: upper bound of bin k (ascending; last = +inf)
};

int to_bin(const Feat& f, float v) {
  return static_cast<int>(std::lower_bound(f.ub.begin(), f.ub.end(), v) - f.ub.begin());
}

// Bins for one column from its sorted non-zero values. Zero (absent) gets its own bin; negative and
// positive values are cut into equal-count bins (each distinct value its own bin when they fit),
// bins share max_bin - 1 slots in proportion to their counts.
void make_bins(std::vector<float>& vals, int max_bin, Feat* out) {
  std::sort(vals.begin(), vals.end());
  std::vector<float> dv;
  std::vector<int64_t> dc;
  for (float v : vals) {
    if (!dv.empty() && v == dv.back()) {
      ++dc.back();
    } else {
      dv.push_back(v);
      dc.push_back(1);
    }
  }
  const auto split =
      std::lower_bound(dv.begin(), dv.end(), 0.0f) - dv.begin();  // dv[0:split) < 0 < dv[split:)
  struct Range {
    float lo, hi;
  };
  std::vector<Range> neg_bins, pos_bins;
  auto cut = [&](int64_t a, int64_t b, int k,
                 std::vector<Range>* bins) {  // distinct values [a, b) into <= k bins
    if (b <= a || k <= 0) return;
    if (b - a <= k) {
      for (int64_t i = a; i < b; ++i) bins->push_back({dv[i], dv[i]});
      return;
    }
    int64_t total = 0;
    for (int64_t i = a; i < b; ++i) total += dc[i];
    double target = static_cast<double>(total) / k;
    int64_t acc = 0;
    int64_t start = a;
    int made = 0;
    for (int64_t i = a; i < b; ++i) {
      acc += dc[i];
      const int64_t left_distinct = b - i - 1;
      const int left_bins = k - made - 1;
      if ((acc >= target && left_bins > 0 && left_distinct >= left_bins) || i == b - 1) {
        bins->push_back({dv[start], dv[i]});
        ++made;
        start = i + 1;
        if (left_bins > 0) {
          int64_t rest = 0;
          for (int64_t j = i + 1; j < b; ++j) rest += dc[j];
          target = static_cast<double>(rest) / left_bins;
        }
        acc = 0;
      }
    }
  };
  int64_t cnt_neg = 0, cnt_pos = 0;
  for (int64_t i = 0; i < split; ++i) cnt_neg += dc[i];
  for (int64_t i = split; i < static_cast<int64_t>(dv.size()); ++i) cnt_pos += dc[i];
  const int avail = max_bin - 1;
  const int64_t n_neg = split, n_pos = static_cast<int64_t>(dv.size()) - split;
  int k_neg = 0, k_pos = 0;
  if (n_neg + n_pos <= avail) {
    k_neg = static_cast<int>(n_neg);
    k_pos = static_cast<int>(n_pos);
  } else {
    k_neg = n_neg == 0 ? 0
                       : std::max(1, static_cast<int>(std::lround(avail * double(cnt_neg) /
                                                                  (cnt_neg + cnt_pos))));
    k_neg = static_cast<int>(std::min<int64_t>(k_neg, n_neg));
    k_pos = static_cast<int>(std::min<int64_t>(avail - k_neg, n_pos));
    k_neg = static_cast<int>(std::min<int64_t>(avail - k_pos, n_neg));
  }
  cut(0, split, k_neg, &neg_bins);
  cut(split, static_cast<int64_t>(dv.size()), k_pos, &pos_bins);
  std::vector<Range> bins = neg_bins;
  out->zero_bin = static_cast<int>(bins.size());
  bins.push_back({0.0f, 0.0f});
  bins.insert(bins.end(), pos_bins.begin(), pos_bins.end());
  out->nbin = static_cast<int>(bins.size());
  out->ub.resize(bins.size());
  for (size_t k = 0; k + 1 < bins.size(); ++k) out->ub[k] = 0.5f * (bins[k].hi + bins[k + 1].lo);
  out->ub.back() = std::numeric_limits<float>::infinity();
}

#ifdef SGB_F32
struct HB {  // histogram bin (single precision: 12 bytes)
  float g, h;
  int32_t c;
};
#else
struct HB {  // histogram bin
  double g, h;
  int64_t c;
};
#endif

// Allocator that leaves new elements uninitialized on resize: histogram buffers are recycled across
// splits and layouts, and every bin is written before it is read.
template <class T>
struct NoInit : std::allocator<T> {
  template <class U>
  struct rebind {
    using other = NoInit<U>;
  };
  NoInit() = default;
  template <class U>
  NoInit(const NoInit<U>&) noexcept {}
  template <class U>
  void construct(U* p) noexcept {
    ::new (static_cast<void*>(p)) U;
  }
  template <class U, class... A>
  void construct(U* p, A&&... a) {
    ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
  }
};
using HBVec = std::vector<HB, NoInit<HB>>;

struct Split {
  double gain = -1.0;  // > 0 when valid
  int feat = -1;       // local feature id (in the tree's sampled set)
  int thr = 0;         // left = bin <= thr
  double gl = 0, hl = 0;
  int64_t cl = 0;
};

struct Node {  // tree node; leaves are encoded as ~leaf_index in children
  int feat;    // global feature id
  int thr;
  int left, right;
};

struct Tree {
  std::vector<Node> nodes;
  std::vector<double> leaf_value;
};

struct RowBins {  // row-major (feature, bin) entries, sorted by feature within each row
  std::vector<int64_t> ptr;
  std::vector<int32_t> feat;
  std::vector<uint8_t> bin;
  int bin_of(int64_t r, int f, int zero_bin) const {
    const int32_t* a = feat.data() + ptr[r];
    const int32_t* b = feat.data() + ptr[r + 1];
    const int32_t* p = std::lower_bound(a, b, f);
    return (p != b && *p == f) ? bin[p - feat.data()] : zero_bin;
  }
};

int route(const Tree& t, const RowBins& rb, int64_t r, const std::vector<Feat>& feats) {
  int node = 0;
  while (node >= 0) {
    const Node& n = t.nodes[node];
    node = rb.bin_of(r, n.feat, feats[n.feat].zero_bin) <= n.thr ? n.left : n.right;
  }
  return ~node;
}

// CSR input (float values) -> RowBins keeping only non-zero-bin entries.
RowBins bin_csr(int64_t n, const int64_t* ptr, const int32_t* idx, const float* val,
                const std::vector<Feat>& feats) {
  RowBins rb;
  rb.ptr.assign(n + 1, 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (int64_t r = 0; r < n; ++r) {
    int64_t k = 0;
    for (int64_t j = ptr[r]; j < ptr[r + 1]; ++j) {
      const Feat& f = feats[idx[j]];
      if (f.nbin >= 2 && to_bin(f, val[j]) != f.zero_bin) ++k;
    }
    rb.ptr[r + 1] = k;
  }
  for (int64_t r = 0; r < n; ++r) rb.ptr[r + 1] += rb.ptr[r];
  rb.feat.resize(rb.ptr[n]);
  rb.bin.resize(rb.ptr[n]);
#pragma omp parallel for schedule(dynamic, 256)
  for (int64_t r = 0; r < n; ++r) {
    int64_t o = rb.ptr[r];
    for (int64_t j = ptr[r]; j < ptr[r + 1]; ++j) {
      const Feat& f = feats[idx[j]];
      if (f.nbin < 2) continue;
      const int b = to_bin(f, val[j]);
      if (b != f.zero_bin) {
        rb.feat[o] = idx[j];
        rb.bin[o] = static_cast<uint8_t>(b);
        ++o;
      }
    }
  }
  return rb;
}

double logloss(const std::vector<double>& s, const float* y) {
  double t = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const double p = 1.0 / (1.0 + std::exp(-s[i]));
    t -= y[i] > 0.5f ? std::log(std::max(p, 1e-15)) : std::log(std::max(1 - p, 1e-15));
  }
  return t / s.size();
}

// A trained model: init score, trees (up to the best iteration), and the bins of the features the
// trees use.
struct Model {
  int64_t m_cols = 0;
  double init = 0;
  std::vector<Tree> trees;
  std::vector<int> fpos;  // global feature -> index in fs (-1 if unused)
  std::vector<Feat> fs;   // bins of used features
  std::vector<int> fids;  // global feature ids of fs
  void index() {
    fpos.assign(m_cols, -1);
    for (size_t i = 0; i < fids.size(); ++i) fpos[fids[i]] = static_cast<int>(i);
  }
};

// Raw score of one CSR row (indices sorted ascending).
double predict_row(const Model& m, const int32_t* idx, const float* val, int64_t len) {
  double s = m.init;
  for (const Tree& t : m.trees) {
    if (t.leaf_value.size() == 1) {
      s += t.leaf_value[0];
      continue;
    }
    int node = 0;
    while (node >= 0) {
      const Node& tn = t.nodes[node];
      const Feat& f = m.fs[m.fpos[tn.feat]];
      const int32_t* p = std::lower_bound(idx, idx + len, tn.feat);
      const int bin = (p != idx + len && *p == tn.feat) ? to_bin(f, val[p - idx]) : f.zero_bin;
      node = bin <= tn.thr ? tn.left : tn.right;
    }
    s += t.leaf_value[~node];
  }
  return s;
}

}  // namespace

extern "C" {

// params: [0] threads, [1] max_bin, [2] num_leaves, [3] min_child_samples, [4] lambda_l2,
//         [5] learning_rate, [6] bagging_fraction, [7] max_rounds, [8] early_stopping, [9] seed,
//         [10] min_sum_hessian, [11] slot_k, [12] refresh, [13] unused, [14] layouts,
//         [15] goss_top, [16] goss_other
// info out: [0] best_iter, [1] trees, [2] t_bins, [3] t_rowbins, [4] t_trees_total, [5] t_subset,
//           [6] t_grow, [7] t_score, [8] train_rows_nnz_kept, [9] t_root, [10] t_partition,
//           [11] t_big_splits, [12] t_small_splits, [13] n_big, [14] n_small
void* sgb_train(int64_t n, int64_t m_cols, const int64_t* csc_ptr, const int32_t* csc_idx,
                const float* csc_val, const float* y, int64_t n_tun, const int64_t* tun_ptr,
                const int32_t* tun_idx, const float* tun_val, const float* y_tun, int64_t n_held,
                const int64_t* held_ptr, const int32_t* held_idx, const float* held_val,
                const int32_t* slot_of, const double* params, double* out_tun, double* out_held,
                double* info) {
  const int threads = static_cast<int>(params[0]);
  const int max_bin = std::min(255, static_cast<int>(params[1]));
  const int num_leaves = static_cast<int>(params[2]);
  const int64_t min_child = static_cast<int64_t>(params[3]);
  const double lambda = params[4], lr = params[5], bag = params[6];
  const int max_rounds = static_cast<int>(params[7]), patience = static_cast<int>(params[8]);
  const uint64_t seed = static_cast<uint64_t>(params[9]);
  const double min_hess = params[10];
  const int slot_k = std::max(1, static_cast<int>(params[11]));
  const int refresh = std::max(1, static_cast<int>(params[12]));  // trees per feature sample
  const double goss_top = params[15],
               goss_other = params[16];  // GOSS row sampling (goss_top = 0: bagging)
  const int n_layouts =
      std::max(0, static_cast<int>(params[14]));  // precomputed layouts (0 = sample per refresh)
  omp_set_num_threads(threads);

  // 1. Bins per column (parallel), from the non-zero training values.
  auto t0 = Clock::now();
  std::vector<Feat> feats(m_cols);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t c = 0; c < m_cols; ++c) {
    std::vector<float> v;
    v.reserve(csc_ptr[c + 1] - csc_ptr[c]);
    for (int64_t j = csc_ptr[c]; j < csc_ptr[c + 1]; ++j)
      if (csc_val[j] != 0.0f) v.push_back(csc_val[j]);
    if (v.empty()) {
      feats[c].nbin = 1;
      feats[c].zero_bin = 0;
      feats[c].ub = {std::numeric_limits<float>::infinity()};
    } else {
      make_bins(v, max_bin, &feats[c]);
    }
  }
  info[2] = secs(t0);

  // 2. Row-major bins for train / tuning / held-out (non-zero-bin entries only).
  t0 = Clock::now();
  std::vector<uint8_t> col_bins(csc_ptr[m_cols]);
  int64_t kept = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : kept)
  for (int64_t c = 0; c < m_cols; ++c) {
    const Feat& fc = feats[c];
    for (int64_t j = csc_ptr[c]; j < csc_ptr[c + 1]; ++j) {
      const int k = fc.nbin >= 2 ? to_bin(fc, csc_val[j]) : fc.zero_bin;
      col_bins[j] = (k == fc.zero_bin) ? 255 : static_cast<uint8_t>(k);
      kept += k != fc.zero_bin;
    }
  }
  const RowBins tu = bin_csr(n_tun, tun_ptr, tun_idx, tun_val, feats);
  const RowBins ho = bin_csr(n_held, held_ptr, held_idx, held_val, feats);
  info[3] = secs(t0);
  info[8] = static_cast<double>(kept);

  // Slots: usable features grouped by slot; slot -1 = always used.
  std::vector<int> always;
  std::vector<std::vector<int>> slots;
  {
    std::vector<int> slot_index;
    int max_slot = -1;
    for (int64_t c = 0; c < m_cols; ++c) max_slot = std::max(max_slot, slot_of[c]);
    slot_index.assign(max_slot + 1, -1);
    for (int64_t c = 0; c < m_cols; ++c) {
      if (feats[c].nbin < 2) continue;
      const int s = slot_of[c];
      if (s < 0) {
        always.push_back(static_cast<int>(c));
      } else {
        if (slot_index[s] < 0) {
          slot_index[s] = static_cast<int>(slots.size());
          slots.emplace_back();
        }
        slots[slot_index[s]].push_back(static_cast<int>(c));
      }
    }
  }

  // Boosting state.
  double pos = 0;
  for (int64_t i = 0; i < n; ++i) pos += y[i];
  const double p0 = std::min(std::max(pos / n, 1e-12), 1 - 1e-12);
  const double init = std::log(p0 / (1 - p0));
  std::vector<double> score(n, init), s_tun(n_tun, init), s_held(n_held, init);
  std::vector<double> best_tun(s_tun), best_held(s_held);
  double best_loss = logloss(s_tun, y_tun);
  int best_iter = 0, trees = 0;
  std::vector<Tree> all_trees;
  std::mt19937_64 rng(seed);
  std::vector<double> g(n), h(n), gh(2 * n);
  std::vector<int> used, loc(m_cols, -1);
  std::vector<int64_t> off, sp;
  std::vector<uint16_t> sv;  // row-major entries: bin id local to its chunk (bin - off[cf[c]])
  int L = 0;
  int64_t total_bins = 0;
  std::vector<uint8_t> flag(n, 0);
  std::vector<uint8_t> in_bag(n, 0);
  std::vector<uint8_t> dir;  // partition scratch: 1 = left
  // Dense per-row bins (255 = zero bin) of features already split on, filled on their first split
  // and reused by later splits on them (any tree): a split's partition is then one lookup per row.
  constexpr int64_t kDenseBytes = int64_t{256} << 20;
  const int64_t max_dense = std::min<int64_t>(m_cols, kDenseBytes / std::max<int64_t>(n, 1));
  std::vector<int32_t> dense_of(m_cols, -1);
  std::vector<std::vector<uint8_t>> dense;
  int NC = threads;                // chunks in the current layout (cf.size() - 1)
  std::vector<int> cf(NC + 1, 0);  // chunk c covers local features [cf[c], cf[c+1])
  std::vector<uint32_t> rco;  // rco[r*(NC+1)+c]: first entry of row r in chunk c (absolute in sv)
  double t_subset = 0, t_grow = 0, t_score = 0, t_root = 0, t_part = 0, t_big = 0, t_small = 0;
  int64_t n_big = 0, n_small = 0;
  auto t_trees = Clock::now();
  std::vector<HBVec> pool;  // histogram buffers, reused across splits and trees

  struct Layout {
    std::vector<int> used, loc, cf;
    std::vector<int64_t> off, sp;
    std::vector<uint16_t> sv;
    std::vector<uint32_t> rco;
    int L = 0;
    int64_t total_bins = 0;
  };
  std::vector<Layout> layouts;
  int cur_layout = -1;
  auto swap_with = [&](Layout& lay) {
    used.swap(lay.used);
    loc.swap(lay.loc);
    cf.swap(lay.cf);
    off.swap(lay.off);
    sp.swap(lay.sp);
    sv.swap(lay.sv);
    rco.swap(lay.rco);
    NC = std::max<int>(1, static_cast<int>(cf.size()) - 1);
    std::swap(L, lay.L);
    std::swap(total_bins, lay.total_bins);
  };
  // Subset (sp, sv), chunks (cf) and row chunk offsets (rco) for `used`.
  auto build_current = [&]() {
    L = static_cast<int>(used.size());
    std::fill(loc.begin(), loc.end(), -1);
    off.assign(L + 1, 0);
    for (int l = 0; l < L; ++l) {
      loc[used[l]] = l;
      off[l + 1] = off[l] + feats[used[l]].nbin;
    }
    total_bins = off[L];
    // Chunks: one per thread, balanced by non-zeros, plus a cut wherever a chunk would exceed 65535
    // bins (sv ids are 16-bit and local to their chunk).
    const int T = threads;
    int64_t tot = 0;
    for (int l = 0; l < L; ++l) tot += csc_ptr[used[l] + 1] - csc_ptr[used[l]];
    cf.assign(1, 0);
    {
      int64_t acc = 0;
      int t = 1;
      for (int l = 0; l < L; ++l) {
        if (off[l + 1] - off[cf.back()] > 65535) cf.push_back(l);
        acc += csc_ptr[used[l] + 1] - csc_ptr[used[l]];
        while (t < T && acc >= tot * t / T) {
          cf.push_back(l + 1);
          ++t;
        }
      }
      while (t++ < T) cf.push_back(L);
      if (cf.back() != L) cf.push_back(L);
    }
    NC = static_cast<int>(cf.size()) - 1;
    // Per-(chunk, row) counts, then row starts (sp) and row chunk offsets (rco); entries ascend by
    // local feature within each row.
    std::vector<int64_t> cnt(static_cast<size_t>(NC) * n, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < NC; ++c) {
      int64_t* ct = cnt.data() + static_cast<size_t>(c) * n;
      for (int l = cf[c]; l < cf[c + 1]; ++l)
        for (int64_t j = csc_ptr[used[l]]; j < csc_ptr[used[l] + 1]; ++j)
          if (col_bins[j] != 255) ++ct[csc_idx[j]];
    }
    sp.assign(n + 1, 0);
    for (int64_t r = 0; r < n; ++r) {
      int64_t rt = 0;
      for (int c = 0; c < NC; ++c) {
        const int64_t v = cnt[static_cast<size_t>(c) * n + r];
        cnt[static_cast<size_t>(c) * n + r] = rt;
        rt += v;
      }
      sp[r + 1] = rt;
    }
    for (int64_t r = 0; r < n; ++r) sp[r + 1] += sp[r];
    rco.resize(static_cast<size_t>(n) * (NC + 1));
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n; ++r) {
      uint32_t* ro = rco.data() + static_cast<size_t>(r) * (NC + 1);
      for (int c = 0; c < NC; ++c)
        ro[c] = static_cast<uint32_t>(sp[r] + cnt[static_cast<size_t>(c) * n + r]);
      ro[NC] = static_cast<uint32_t>(sp[r + 1]);
    }
    sv.resize(sp[n]);
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < NC; ++c) {
      // Row-blocked scatter: walk the chunk's columns block by block (columns are sorted by row),
      // so writes stay within a cache-sized window of sv.
      int64_t* pt = cnt.data() + static_cast<size_t>(c) * n;
      const int la = cf[c], lb = cf[c + 1];
      const int64_t base = off[la];
      std::vector<int64_t> cur(lb - la);
      for (int l = la; l < lb; ++l) cur[l - la] = csc_ptr[used[l]];
      constexpr int64_t kRowBlock = 1024;
      for (int64_t r0 = 0; r0 < n; r0 += kRowBlock) {
        const int64_t r1 = std::min(n, r0 + kRowBlock);
        for (int l = la; l < lb; ++l) {
          int64_t j = cur[l - la];
          const int64_t je = csc_ptr[used[l] + 1];
          for (; j < je && csc_idx[j] < r1; ++j)
            if (col_bins[j] != 255) {
              const int64_t r = csc_idx[j];
              sv[sp[r] + pt[r]++] = static_cast<uint16_t>(off[l] - base + col_bins[j]);
            }
          cur[l - la] = j;
        }
      }
    }
  };
  for (int it = 0; it < max_rounds; ++it) {
    auto ts = Clock::now();
    // Gradients.
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      const double p = 1.0 / (1.0 + std::exp(-score[i]));
      g[i] = p - y[i];
      h[i] = std::max(p * (1 - p), 1e-16);
      gh[2 * i] = g[i];
      gh[2 * i + 1] = h[i];
    }
    // Row sampling: GOSS (top |gradient| rows + a reweighted random share of the rest) or bagging
    // (selection sampling). Rows stay in ascending order.
    std::vector<int64_t> rows;
    rows.reserve(static_cast<size_t>(bag * n) + 1);
    if (goss_top > 0) {
      std::vector<double> a(n);
      for (int64_t i = 0; i < n; ++i) a[i] = std::abs(g[i]);
      const int64_t k_top = std::max<int64_t>(1, static_cast<int64_t>(goss_top * n));
      std::vector<double> tmp_a(a);
      std::nth_element(tmp_a.begin(), tmp_a.begin() + (n - k_top), tmp_a.end());
      const double thr = tmp_a[n - k_top];
      const double keep = goss_other / std::max(1e-12, 1.0 - goss_top);  // share of the rest kept
      const double w = (1.0 - goss_top) / std::max(1e-12, goss_other);   // their weight
      std::uniform_real_distribution<double> u(0.0, 1.0);
      for (int64_t i = 0; i < n; ++i) {
        if (a[i] >= thr) {
          rows.push_back(i);
        } else if (u(rng) < keep) {
          rows.push_back(i);
          g[i] *= w;
          h[i] *= w;
          gh[2 * i] = g[i];
          gh[2 * i + 1] = h[i];
        }
      }
    } else {
      int64_t need = static_cast<int64_t>(std::llround(bag * n));
      std::uniform_real_distribution<double> u(0.0, 1.0);
      for (int64_t i = 0; i < n && need > 0; ++i) {
        if (u(rng) * (n - i) < need) {
          rows.push_back(i);
          --need;
        }
      }
    }
    const int64_t mr = static_cast<int64_t>(rows.size());
    std::fill(in_bag.begin(), in_bag.end(), 0);
    for (int64_t r : rows) in_bag[r] = 1;
    if (it % refresh == 0 && n_layouts > 0) {
      if (layouts.empty()) {  // build all layouts once
        layouts.resize(n_layouts);
        std::vector<size_t> base(slots.size(), 0);
        for (int k = 0; k < n_layouts; ++k) {
          if (k % 7 == 0)
            for (size_t si = 0; si < slots.size(); ++si) {
              std::uniform_int_distribution<size_t> pick(0, slots[si].size() - 1);
              base[si] = pick(rng);
            }
          used = always;
          for (size_t si = 0; si < slots.size(); ++si)
            used.push_back(slots[si][(base[si] + k % 7) % slots[si].size()]);
          std::sort(used.begin(), used.end());
          used.erase(std::unique(used.begin(), used.end()), used.end());
          loc.assign(m_cols, -1);
          build_current();
          swap_with(layouts[k]);
        }
      }
      std::uniform_int_distribution<int> pick_layout(0, n_layouts - 1);
      const int j = pick_layout(rng);
      if (cur_layout >= 0) swap_with(layouts[cur_layout]);  // return the current one to its slot
      swap_with(layouts[j]);
      cur_layout = j;
    } else if (it % refresh == 0) {
      // Feature sampling: always-used + slot_k per slot; local ids follow global order.
      used = always;
      for (const auto& sl : slots) {
        if (static_cast<int>(sl.size()) <= slot_k) {
          used.insert(used.end(), sl.begin(), sl.end());
        } else {
          for (int k = 0; k < slot_k; ++k) {
            std::uniform_int_distribution<size_t> pick(0, sl.size() - 1);
            used.push_back(sl[pick(rng)]);  // with replacement for k > 1 (dups removed below)
          }
        }
      }
      std::sort(used.begin(), used.end());
      used.erase(std::unique(used.begin(), used.end()), used.end());
      build_current();
    }
    const std::vector<double>& gs = g;
    const std::vector<double>& hs = h;
    t_subset += secs(ts);

    // Grow one tree, leaf-wise.
    ts = Clock::now();
    struct Leaf {
      int64_t begin, end;
      double G, H;
      int64_t C;
      HBVec hist;
      Split best;
      int parent_node;  // node index whose child pointer leads here (-1 for root)
      bool is_left;
      std::vector<int>
          active;  // local features with >= min_child non-zero rows here (only ones that can split)
    };
    std::vector<Leaf> leaves;
    std::vector<int64_t> order(rows);
    auto get_hist =
        [&]() {  // buffers are reused across splits and trees (contents are always overwritten)
          HBVec v;
          if (!pool.empty()) {
            v = std::move(pool.back());
            pool.pop_back();
          }
          v.resize(total_bins);
          return v;
        };
    // Best split of one feature (local l) given node totals; returns the feature's non-zero row
    // count in the node. One side of any split holds only non-zero rows, so fewer than min_child of
    // them means no valid split here or in any descendant (exact pruning).
    auto best_of_feature = [&](const HB* hist, int l, double G, double H, int64_t C,
                               Split* best) -> int64_t {
      const Feat& f = feats[used[l]];
      const HB* hb = hist + off[l];
      double sg = 0, sh = 0;
      int64_t sc = 0;
      for (int b = 0; b < f.nbin; ++b)
        if (b != f.zero_bin) {
          sg += hb[b].g;
          sh += hb[b].h;
          sc += hb[b].c;
        }
      if (sc < min_child) return sc;
      const double parent_gain = G * G / (H + lambda);
      double gl = 0, hl = 0;
      int64_t cl = 0;
      for (int b = 0; b + 1 < f.nbin; ++b) {
        if (b == f.zero_bin) {
          gl += G - sg;
          hl += H - sh;
          cl += C - sc;
        } else {
          gl += hb[b].g;
          hl += hb[b].h;
          cl += hb[b].c;
        }
        const int64_t cr = C - cl;
        const double hr = H - hl;
        if (cl < min_child || cr < min_child || hl < min_hess || hr < min_hess) continue;
        const double gr = G - gl;
        const double gain = gl * gl / (hl + lambda) + gr * gr / (hr + lambda) - parent_gain;
        if (gain > best->gain) *best = Split{gain, l, b, gl, hl, cl};
      }
      return sc;
    };
    // Root.
    auto tr0 = Clock::now();
    {
      Leaf root;
      root.begin = 0;
      root.end = mr;
      root.G = 0;
      root.H = 0;
      for (int64_t i : order) {
        root.G += gs[i];
        root.H += hs[i];
      }
      root.C = mr;
      root.hist = get_hist();
      root.parent_node = -1;
      root.is_left = false;
      // Root histogram by row streaming: chunk c's thread walks the bagged rows in ascending order
      // and reads each row's slice of chunk c (rco), so reads are sequential and per-bin sums keep
      // the row order.
      std::vector<Split> cb(NC);
      std::vector<std::vector<int>> cact(NC);
      const int64_t* ord0 = order.data();
#pragma omp parallel for schedule(dynamic, 1)
      for (int c = 0; c < NC; ++c) {
        if (cf[c + 1] <= cf[c]) continue;
        const int64_t blo = off[cf[c]], bhi = off[cf[c + 1]];
        HB* hist = root.hist.data();
        HB* hc = hist + blo;  // the chunk's slice (sv ids are local to it)
        std::memset(static_cast<void*>(hist + blo), 0, sizeof(HB) * (bhi - blo));
        for (int64_t k = 0; k < mr; ++k) {
          if (k + 16 < mr)
            __builtin_prefetch(rco.data() + static_cast<size_t>(ord0[k + 16]) * (NC + 1) + c);
          if (k + 8 < mr) {
            const uint32_t* ro8 = rco.data() + static_cast<size_t>(ord0[k + 8]) * (NC + 1);
            __builtin_prefetch(sv.data() + ro8[c]);
          }
          const int64_t i = ord0[k];
          const uint32_t* ro = rco.data() + static_cast<size_t>(i) * (NC + 1);
          const uint16_t* p = sv.data() + ro[c];
          const uint16_t* e = sv.data() + ro[c + 1];
          const double gi = gh[2 * i], hi = gh[2 * i + 1];
          for (; p < e; ++p) {
            hc[*p].g += gi;
            hc[*p].h += hi;
            hc[*p].c += 1;
          }
        }
        for (int l = cf[c]; l < cf[c + 1]; ++l)
          if (best_of_feature(hist, l, root.G, root.H, root.C, &cb[c]) >= min_child)
            cact[c].push_back(l);
      }
      for (int c = 0; c < NC; ++c) {
        if (cb[c].gain > root.best.gain) root.best = cb[c];
        root.active.insert(root.active.end(), cact[c].begin(), cact[c].end());
      }
      leaves.push_back(std::move(root));
    }
    t_root += secs(tr0);
    Tree tree;
    std::vector<int64_t> tmp(mr);
    while (static_cast<int>(leaves.size()) < num_leaves) {
      int pick = -1;
      double bg = 1e-12;
      for (int i = 0; i < static_cast<int>(leaves.size()); ++i)
        if (leaves[i].best.gain > bg) {
          bg = leaves[i].best.gain;
          pick = i;
        }
      if (pick < 0) break;
      Leaf& P = leaves[pick];
      const Split s = P.best;
      const int l = s.feat;
      const Feat& f = feats[used[l]];
      auto tp0 = Clock::now();
      // Partition P's rows (stable): left = bin <= thr.
      int64_t nl = 0, nr = 0;
      const int gf = used[l];
      const int64_t col_nnz = csc_ptr[gf + 1] - csc_ptr[gf];
      const int64_t np = P.end - P.begin;
      if (static_cast<int64_t>(dir.size()) < np) dir.resize(np);
      const bool par = np >= 16384;
      const bool col_par = col_nnz >= 16384;
      const bool by_column = np * 8 > col_nnz;
      const int lc = static_cast<int>(std::upper_bound(cf.begin(), cf.end(), l) - cf.begin()) -
                     1;  // l's chunk
      const uint16_t flo = static_cast<uint16_t>(off[l] - off[cf[lc]]),
                     fhi = static_cast<uint16_t>(off[l + 1] - off[cf[lc]]);  // chunk-local
      if (dense_of[gf] < 0 && static_cast<int64_t>(dense.size()) < max_dense) {
        dense_of[gf] = static_cast<int32_t>(dense.size());
        dense.emplace_back(n, uint8_t{255});
        uint8_t* d = dense.back().data();
#pragma omp parallel for schedule(static) if (col_par)
        for (int64_t j = csc_ptr[gf]; j < csc_ptr[gf + 1]; ++j) d[csc_idx[j]] = col_bins[j];
      }
      const bool dense_inline = dense_of[gf] >= 0 && np < 65536;
      if (dense_inline) {              // small node: decide and place in one serial pass (below)
      } else if (dense_of[gf] >= 0) {  // one lookup per row
        const uint8_t* d = dense[dense_of[gf]].data();
        const bool zero_left = f.zero_bin <= s.thr;
        const int thr = s.thr;
#pragma omp parallel for schedule(static) if (par)
        for (int64_t k = 0; k < np; ++k) {
          const uint8_t v = d[order[P.begin + k]];
          dir[k] = (v == 255 ? zero_left : v <= thr) ? 1 : 0;
        }
      } else if (by_column) {  // flag rows from the column, then a sequential lookup per row
#pragma omp parallel for schedule(static) if (col_par)
        for (int64_t j = csc_ptr[gf]; j < csc_ptr[gf + 1]; ++j)
          if (col_bins[j] != 255) flag[csc_idx[j]] = col_bins[j] <= s.thr ? 1 : 2;
        const bool zero_left = f.zero_bin <= s.thr;
#pragma omp parallel for schedule(static) if (par)
        for (int64_t k = 0; k < np; ++k) {
          const uint8_t fl = flag[order[P.begin + k]];
          dir[k] = (fl == 0 ? zero_left : fl == 1) ? 1 : 0;
        }
#pragma omp parallel for schedule(static) if (col_par)
        for (int64_t j = csc_ptr[gf]; j < csc_ptr[gf + 1]; ++j) flag[csc_idx[j]] = 0;
      } else {
#pragma omp parallel for schedule(static) if (par)
        for (int64_t k = 0; k < np; ++k) {
          if (k + 16 < np)
            __builtin_prefetch(rco.data() +
                               static_cast<size_t>(order[P.begin + k + 16]) * (NC + 1) + lc);
          if (k + 8 < np) {
            const int64_t i8 = order[P.begin + k + 8];
            __builtin_prefetch(sv.data() + rco[static_cast<size_t>(i8) * (NC + 1) + lc]);
          }
          const int64_t i = order[P.begin + k];
          // Search only the row's slice of the feature's chunk.
          const uint32_t* ro = rco.data() + static_cast<size_t>(i) * (NC + 1);
          const uint16_t* a = sv.data() + ro[lc];
          const uint16_t* b = sv.data() + ro[lc + 1];
          const uint16_t* q = std::lower_bound(a, b, flo);
          const int bin = (q != b && *q < fhi) ? static_cast<int>(*q - flo) : f.zero_bin;
          dir[k] = bin <= s.thr ? 1 : 0;
        }
      }
      if (dense_inline) {
        const uint8_t* d = dense[dense_of[gf]].data();
        const bool zero_left = f.zero_bin <= s.thr;
        const int thr = s.thr;
        for (int64_t k = 0; k < np; ++k) {  // stable placement: left rows in place, right via tmp
          const int64_t i = order[P.begin + k];
          const uint8_t v = d[i];
          if (v == 255 ? zero_left : v <= thr) {
            order[P.begin + nl++] = i;
          } else {
            tmp[nr++] = i;
          }
        }
      } else {
        for (int64_t k = 0; k < np;
             ++k) {  // stable placement: left rows in place, right rows via tmp
          const int64_t i = order[P.begin + k];
          if (dir[k]) {
            order[P.begin + nl++] = i;
          } else {
            tmp[nr++] = i;
          }
        }
      }
      std::copy(tmp.begin(), tmp.begin() + nr, order.begin() + P.begin + nl);
      // Record the split in the tree.
      const int node_id = static_cast<int>(tree.nodes.size());
      tree.nodes.push_back(Node{used[l], s.thr, 0, 0});
      if (P.parent_node >= 0) {
        if (P.is_left)
          tree.nodes[P.parent_node].left = node_id;
        else
          tree.nodes[P.parent_node].right = node_id;
      }
      Leaf Lc, Rc;
      Lc.begin = P.begin;
      Lc.end = P.begin + nl;
      Lc.G = s.gl;
      Lc.H = s.hl;
      Lc.C = s.cl;
      Rc.begin = P.begin + nl;
      Rc.end = P.end;
      Rc.G = P.G - s.gl;
      Rc.H = P.H - s.hl;
      Rc.C = P.C - s.cl;
      Lc.parent_node = Rc.parent_node = node_id;
      Lc.is_left = true;
      Rc.is_left = false;
      const bool left_small = (Lc.end - Lc.begin) <= (Rc.end - Rc.begin);
      Leaf& S = left_small ? Lc : Rc;
      Leaf& B = left_small ? Rc : Lc;
      S.hist = get_hist();
      B.hist = std::move(P.hist);
      const std::vector<int> act = std::move(P.active);
      t_part += secs(tp0);
      auto tsr = Clock::now();
      const int64_t* ord = order.data();
      const bool parallel =
          act.size() >= 256;  // enough per-feature work to repay one synchronisation
      std::vector<Split> cs(NC), cbig(NC);
      std::vector<std::vector<int>> as(NC), ab(NC);
      // act is ascending: each chunk's active features are a contiguous sub-range of it.
      std::vector<int> qa(NC + 1);
      for (int c = 0; c <= NC; ++c)
        qa[c] = static_cast<int>(std::lower_bound(act.begin(), act.end(), cf[c]) - act.begin());
#pragma omp parallel for schedule(dynamic, 1) if (parallel)
      for (int c = 0; c < NC; ++c) {
        if (qa[c + 1] <= qa[c]) continue;  // no active feature here: rows are not touched
        HB* hs_ = S.hist.data();
        HB* hb_ = B.hist.data();
        HB* hsc = hs_ + off[cf[c]];  // the chunk's slice (sv ids are local to it)
        for (int q = qa[c]; q < qa[c + 1]; ++q) {
          const int ll = act[q];
          std::memset(static_cast<void*>(hs_ + off[ll]), 0, sizeof(HB) * (off[ll + 1] - off[ll]));
        }
        // Entries of inactive features of this chunk land in never-read slots of the buffer:
        // harmless.
        for (int64_t k = S.begin; k < S.end; ++k) {
          if (k + 16 < S.end)
            __builtin_prefetch(rco.data() + static_cast<size_t>(ord[k + 16]) * (NC + 1) + c);
          if (k + 8 < S.end) {
            const int64_t i8 = ord[k + 8];
            const uint32_t* ro8 = rco.data() + static_cast<size_t>(i8) * (NC + 1);
            __builtin_prefetch(sv.data() + ro8[c]);
            __builtin_prefetch(sv.data() + ro8[c] + 16);
            __builtin_prefetch(gh.data() + 2 * i8);
          }
          const int64_t i = ord[k];
          const uint32_t* ro = rco.data() + static_cast<size_t>(i) * (NC + 1);
          const uint16_t* p = sv.data() + ro[c];
          const uint16_t* b = sv.data() + ro[c + 1];
          const double gi = gh[2 * i], hi = gh[2 * i + 1];
          for (; p < b; ++p) {
            hsc[*p].g += gi;
            hsc[*p].h += hi;
            hsc[*p].c += 1;
          }
        }
        for (int q = qa[c]; q < qa[c + 1]; ++q) {
          const int ll = act[q];
          for (int64_t z = off[ll]; z < off[ll + 1]; ++z) {
            hb_[z].g -= hs_[z].g;
            hb_[z].h -= hs_[z].h;
            hb_[z].c -= hs_[z].c;
          }
          if (best_of_feature(hs_, ll, S.G, S.H, S.C, &cs[c]) >= min_child) as[c].push_back(ll);
          if (best_of_feature(hb_, ll, B.G, B.H, B.C, &cbig[c]) >= min_child) ab[c].push_back(ll);
        }
      }
      for (int c = 0; c < NC; ++c) {
        if (cs[c].gain > S.best.gain) S.best = cs[c];
        if (cbig[c].gain > B.best.gain) B.best = cbig[c];
        S.active.insert(S.active.end(), as[c].begin(), as[c].end());
        B.active.insert(B.active.end(), ab[c].begin(), ab[c].end());
      }
      if (parallel) {
        t_big += secs(tsr);
        ++n_big;
      } else {
        t_small += secs(tsr);
        ++n_small;
      }
      leaves[pick] = std::move(Lc);
      leaves.push_back(std::move(Rc));
    }
    // Leaf values, leaf pointers.
    const int n_leaves = static_cast<int>(leaves.size());
    tree.leaf_value.resize(n_leaves);
    for (int i = 0; i < n_leaves; ++i) {
      Leaf& Lf = leaves[i];
      tree.leaf_value[i] = -lr * Lf.G / (Lf.H + lambda);
      if (Lf.parent_node >= 0) {
        if (Lf.is_left)
          tree.nodes[Lf.parent_node].left = ~i;
        else
          tree.nodes[Lf.parent_node].right = ~i;
      }
    }
    if (tree.nodes.empty()) {  // no valid split: single-leaf tree
      tree.nodes.push_back(Node{0, 0, ~0, ~0});
      tree.leaf_value.assign(1, -lr * leaves[0].G / (leaves[0].H + lambda));
    }
    for (auto& Lf : leaves)
      if (!Lf.hist.empty()) pool.push_back(std::move(Lf.hist));
    t_grow += secs(ts);

    // Scores: all train rows (bagged and not), tuning, held-out.
    ts = Clock::now();
    for (int li = 0; li < static_cast<int>(leaves.size()); ++li)
      for (int64_t k = leaves[li].begin; k < leaves[li].end; ++k)
        score[order[k]] += tree.leaf_value[li];
    if (tree.leaf_value.size() == 1) {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < n; ++i)
        if (!in_bag[i]) score[i] += tree.leaf_value[0];
    } else {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < n; ++i) {
        if (in_bag[i]) continue;
        int node = 0;
        while (node >= 0) {
          const Node& tn = tree.nodes[node];
          const int l = loc[tn.feat];
          const int lc =
              static_cast<int>(std::upper_bound(cf.begin(), cf.end(), l) - cf.begin()) - 1;
          const uint32_t* ro = rco.data() + static_cast<size_t>(i) * (NC + 1);
          const uint16_t flo = static_cast<uint16_t>(off[l] - off[cf[lc]]),
                         fhi = static_cast<uint16_t>(off[l + 1] - off[cf[lc]]);
          const uint16_t* a = sv.data() + ro[lc];
          const uint16_t* b = sv.data() + ro[lc + 1];
          const uint16_t* q = std::lower_bound(a, b, flo);
          const int bin =
              (q != b && *q < fhi) ? static_cast<int>(*q - flo) : feats[tn.feat].zero_bin;
          node = bin <= tn.thr ? tn.left : tn.right;
        }
        score[i] += tree.leaf_value[~node];
      }
    }
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_tun; ++i) s_tun[i] += tree.leaf_value[route(tree, tu, i, feats)];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_held; ++i) s_held[i] += tree.leaf_value[route(tree, ho, i, feats)];
    t_score += secs(ts);
    ++trees;
    all_trees.push_back(std::move(tree));
    const double loss = logloss(s_tun, y_tun);
    if (loss < best_loss) {
      best_loss = loss;
      best_iter = it + 1;
      best_tun = s_tun;
      best_held = s_held;
    } else if (it + 1 - best_iter >= patience) {
      break;
    }
  }
  std::copy(best_tun.begin(), best_tun.end(), out_tun);
  std::copy(best_held.begin(), best_held.end(), out_held);
  info[0] = best_iter;
  info[1] = trees;
  info[4] = secs(t_trees);
  info[5] = t_subset;
  info[6] = t_grow;
  info[7] = t_score;
  info[9] = t_root;
  info[10] = t_part;
  info[11] = t_big;
  info[12] = t_small;
  info[13] = n_big;
  info[14] = n_small;
  Model* model = new Model();
  model->m_cols = m_cols;
  model->init = init;
  all_trees.resize(best_iter);
  model->trees = std::move(all_trees);
  std::vector<char> seen(m_cols, 0);
  for (const Tree& t : model->trees)
    if (t.leaf_value.size() > 1)
      for (const Node& tn : t.nodes) seen[tn.feat] = 1;
  for (int64_t c = 0; c < m_cols; ++c)
    if (seen[c]) {
      model->fids.push_back(static_cast<int>(c));
      model->fs.push_back(feats[c]);
    }
  model->index();
  return model;
}

// Benchmark entry point: train, write tuning / held-out raw scores, free the model.
int sgb_fit(int64_t n, int64_t m_cols, const int64_t* csc_ptr, const int32_t* csc_idx,
            const float* csc_val, const float* y, int64_t n_tun, const int64_t* tun_ptr,
            const int32_t* tun_idx, const float* tun_val, const float* y_tun, int64_t n_held,
            const int64_t* held_ptr, const int32_t* held_idx, const float* held_val,
            const int32_t* slot_of, const double* params, double* out_tun, double* out_held,
            double* info) {
  void* m =
      sgb_train(n, m_cols, csc_ptr, csc_idx, csc_val, y, n_tun, tun_ptr, tun_idx, tun_val, y_tun,
                n_held, held_ptr, held_idx, held_val, slot_of, params, out_tun, out_held, info);
  delete static_cast<Model*>(m);
  return 0;
}

// Raw scores (log-odds) for a CSR matrix with sorted column indices and the training column layout.
int sgb_predict(const void* model, int64_t n, const int64_t* ptr, const int32_t* idx,
                const float* val, int threads, double* out) {
  const Model& m = *static_cast<const Model*>(model);
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads)
  for (int64_t r = 0; r < n; ++r)
    out[r] = predict_row(m, idx + ptr[r], val + ptr[r], ptr[r + 1] - ptr[r]);
  return 0;
}

int64_t sgb_num_trees(const void* model) {
  return static_cast<int64_t>(static_cast<const Model*>(model)->trees.size());
}

int64_t sgb_num_features(const void* model) { return static_cast<const Model*>(model)->m_cols; }

void sgb_free(void* model) { delete static_cast<Model*>(model); }

// Text format: header, init score, features (id, nbin, zero bin, upper bounds), trees (nodes, then
// leaf values).
int sgb_save(const void* model, const char* path) {
  const Model& m = *static_cast<const Model*>(model);
  std::ofstream out(path);
  if (!out) return 1;
  out.precision(17);
  out << "sparsegb 1\n" << "num_columns " << m.m_cols << "\n" << "init " << m.init << "\n";
  out << "features " << m.fids.size() << "\n";
  for (size_t i = 0; i < m.fids.size(); ++i) {
    const Feat& f = m.fs[i];
    out << m.fids[i] << " " << f.nbin << " " << f.zero_bin;
    for (float u : f.ub) {
      if (std::isinf(u))
        out << " inf";
      else
        out << " " << static_cast<double>(u);
    }
    out << "\n";
  }
  out << "trees " << m.trees.size() << "\n";
  for (const Tree& t : m.trees) {
    out << "tree " << t.nodes.size() << " " << t.leaf_value.size() << "\n";
    for (const Node& tn : t.nodes)
      out << tn.feat << " " << tn.thr << " " << tn.left << " " << tn.right << "\n";
    for (size_t i = 0; i < t.leaf_value.size(); ++i) out << (i ? " " : "") << t.leaf_value[i];
    out << "\n";
  }
  return out.good() ? 0 : 1;
}

void* sgb_load(const char* path) {
  std::ifstream in(path);
  if (!in) return nullptr;
  std::string tag;
  int version = 0;
  Model* m = new Model();
  size_t nf = 0, nt = 0;
  if (!(in >> tag >> version) || tag != "sparsegb" || version != 1) {
    delete m;
    return nullptr;
  }
  in >> tag >> m->m_cols >> tag >> m->init >> tag >> nf;
  m->fids.resize(nf);
  m->fs.resize(nf);
  for (size_t i = 0; i < nf; ++i) {
    Feat& f = m->fs[i];
    in >> m->fids[i] >> f.nbin >> f.zero_bin;
    f.ub.resize(f.nbin);
    for (int k = 0; k < f.nbin; ++k) {
      std::string u;
      in >> u;
      f.ub[k] = (u == "inf") ? std::numeric_limits<float>::infinity() : std::stof(u);
    }
  }
  in >> tag >> nt;
  m->trees.resize(nt);
  for (Tree& t : m->trees) {
    size_t nn = 0, nl = 0;
    in >> tag >> nn >> nl;
    t.nodes.resize(nn);
    for (Node& tn : t.nodes) in >> tn.feat >> tn.thr >> tn.left >> tn.right;
    t.leaf_value.resize(nl);
    for (double& v : t.leaf_value) in >> v;
  }
  if (!in) {
    delete m;
    return nullptr;
  }
  m->index();
  return m;
}

}  // extern "C"
