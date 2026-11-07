#include "shard.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "parallel.h"

namespace meds_tab {
namespace {

// Whether rows at `time` drop events at or before time - width. False for full
// history, and when the cutoff would overflow.
bool HasCutoff(std::int64_t time, std::int64_t width) {
  return width != -1 && time >= std::numeric_limits<std::int64_t>::min() + width;
}

void ValidateWindows(const std::vector<std::int64_t>& windows) {
  if (windows.empty()) throw std::invalid_argument("At least one window is required.");
  for (auto width : windows) {
    if (width != -1 && width <= 0) {
      throw std::invalid_argument("Window widths must be positive, or -1 for full history.");
    }
  }
}

void ValidateThreads(int threads) {
  if (threads < 1) throw std::invalid_argument("threads must be positive.");
}

// A compensated (Neumaier) sum: correction keeps the low-order bits that sum
// drops when values of different magnitudes are added.
struct Sum {
  double sum = 0, correction = 0;

  void Add(double value) {
    const double next = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
  }
  double Value() const { return sum + correction; }
};

inline std::uint64_t Bit(std::int32_t code) { return std::uint64_t{1} << (code & 63); }

// One worker's buffers, indexed by subject-local code. They grow to the largest
// subject the worker sees and are then reused, so the event loop never allocates.
struct Scratch {
  std::vector<std::int64_t> records, numbers;
  // Window sums that never subtract, so a value leaving the window cannot disturb
  // the others' sum. A code's values in the window occupy slots [head, tail) of the
  // code's slice, in arrival order. Slots [head, mid) are the front: suffix[k] sums
  // slots k to mid - 1. Slots [mid, tail) are the back, summed in back. The window
  // sum is suffix[head] + back. When the front empties, the back becomes the front
  // by summing its suffixes once, so each value is summed into a suffix only once.
  std::vector<std::int32_t> sum_head, sum_mid, sum_tail;
  std::vector<double> slots;
  std::vector<Sum> suffix_sum, suffix_squares, back_sum, back_squares;
  // Monotonic queues of event offsets within the subject, one per code, each in
  // the code's slice of the subject's code-grouped events.
  std::vector<std::int32_t> min_head, min_tail, max_head, max_tail;
  std::vector<std::int32_t> min_queue, max_queue;
  // Bitsets of codes with records (active) and with values (valued).
  std::vector<std::uint64_t> active, valued;
  // Sizing: per window, the left cursor and the active and valued code counts.
  std::vector<std::int64_t> left, active_count, valued_count;
  // Reordering a row whose value columns are not ascending.
  std::vector<std::pair<std::int64_t, std::int64_t>> row_order;
  std::vector<double> row_values;
};

// The selected numeric statistics, in output order.
struct NumericPlan {
  std::vector<int> stats;
  bool sum = false, squares = false, min = false, max = false;

  explicit NumericPlan(std::uint32_t mask) {
    for (int stat = kValueCount; stat < kStatisticCount; ++stat) {
      if (!(mask & (1u << stat))) continue;
      stats.push_back(stat);
      sum = sum || stat == kSum;
      squares = squares || stat == kSumSquared;
      min = min || stat == kMin;
      max = max || stat == kMax;
    }
  }
};

// Sweeps one subject's events once for all windows and reports each row's stored
// entries as sizes(window, row, code entries, value entries). Code entries are
// codes with a record in the window; value entries are codes with a value.
template <typename Sizes>
void SizeRows(const Events& events, const std::int32_t* local_codes, std::int64_t begin,
              std::int64_t end, std::size_t vocab_size, Rows rows, std::int64_t row_begin,
              std::int64_t row_end, const std::vector<std::int64_t>& windows, bool numeric,
              Scratch& s, Sizes sizes) {
  const auto w_count = windows.size();
  s.records.assign(w_count * vocab_size, 0);
  if (numeric) s.numbers.assign(w_count * vocab_size, 0);
  s.left.assign(w_count, begin);
  s.active_count.assign(w_count, 0);
  s.valued_count.assign(w_count, 0);
  auto right = begin;
  for (auto row = row_begin; row < row_end; ++row) {
    const auto time = rows.times[row];
    for (; right < end && events.times[right] <= time; ++right) {
      const auto code = local_codes[right];
      const bool value = numeric && !std::isnan(events.values[right]);
      for (std::size_t w = 0; w < w_count; ++w) {
        if (s.records[w * vocab_size + code]++ == 0) ++s.active_count[w];
        if (value && s.numbers[w * vocab_size + code]++ == 0) ++s.valued_count[w];
      }
    }
    for (std::size_t w = 0; w < w_count; ++w) {
      if (HasCutoff(time, windows[w])) {
        const auto cutoff = time - windows[w];
        for (auto& left = s.left[w]; left < right && events.times[left] <= cutoff; ++left) {
          const auto code = local_codes[left];
          if (--s.records[w * vocab_size + code] == 0) --s.active_count[w];
          if (numeric && !std::isnan(events.values[left]) &&
              --s.numbers[w * vocab_size + code] == 0) {
            --s.valued_count[w];
          }
        }
      }
      sizes(w, row, s.active_count[w], s.valued_count[w]);
    }
  }
}

}  // namespace

void ValidateRows(Rows rows) {
  if (rows.size < 0) throw std::invalid_argument("Row counts must be nonnegative.");
  if (rows.size && (!rows.subjects || !rows.times)) {
    throw std::invalid_argument("Nonempty rows require subject and time arrays.");
  }
  for (std::int64_t i = 1; i < rows.size; ++i) {
    if (rows.subjects[i] < rows.subjects[i - 1] ||
        (rows.subjects[i] == rows.subjects[i - 1] && rows.times[i] < rows.times[i - 1])) {
      throw std::invalid_argument("Events and rows must be sorted by (subject_id, time).");
    }
  }
}

Shard::Shard(Events events, std::int64_t code_count, int threads)
    : events_(events), code_count_(code_count) {
  ValidateThreads(threads);
  if (events.size < 0 || code_count < 0) {
    throw std::invalid_argument("Event and code counts must be nonnegative.");
  }
  if (code_count > std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1) {
    throw std::invalid_argument("Code IDs must fit in int32.");
  }
  if (events.size && (!events.subjects || !events.times || !events.codes)) {
    throw std::invalid_argument("Nonempty events require subject, time and code arrays.");
  }
  ValidateRows({events.subjects, events.times, events.size});
  for (std::int64_t i = 0; i < events.size; ++i) {
    if (events.codes[i] < 0 || events.codes[i] >= code_count) {
      throw std::invalid_argument("Event code is outside the vocabulary.");
    }
    if (events.values && std::isinf(events.values[i])) {
      throw std::invalid_argument("Infinite values are unsupported.");
    }
  }
  for (std::int64_t i = 0; i < events.size; ++i) {
    if (i == 0 || events.subjects[i] != events.subjects[i - 1]) {
      subjects_.push_back(events.subjects[i]);
      event_begin_.push_back(i);
    }
  }
  event_begin_.push_back(events.size);

  // Per subject, in parallel: its sorted vocabulary, each event's index into it,
  // and where each code's events start in the subject's code-grouped order.
  // Distinct codes are found with a code -> local index table (cleared after each
  // subject) when the vocabulary is small enough, else by sorting.
  const auto subject_count = subjects_.size();
  std::vector<std::vector<std::int32_t>> vocabs(subject_count), bases(subject_count);
  local_codes_.resize(events.size);
  const bool tabled = code_count <= (std::int64_t{1} << 20);
  std::vector<std::vector<std::int32_t>> tables(WorkerCount(threads, subject_count));
  ParallelFor(subject_count, threads, [&](std::size_t worker, std::size_t s) {
    const auto begin = event_begin_[s], end = event_begin_[s + 1];
    if (end - begin > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("A subject exceeds 2^31 - 1 events.");
    }
    auto& codes = vocabs[s];
    auto& table = tables[worker];
    if (tabled) {
      if (table.empty()) table.assign(code_count, -1);
      for (auto i = begin; i < end; ++i) {
        auto& slot = table[events.codes[i]];
        if (slot < 0) {
          slot = 0;
          codes.push_back(events.codes[i]);
        }
      }
      std::sort(codes.begin(), codes.end());
      for (std::size_t k = 0; k < codes.size(); ++k) table[codes[k]] = static_cast<std::int32_t>(k);
    } else {
      codes.assign(events.codes + begin, events.codes + end);
      std::sort(codes.begin(), codes.end());
      codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
    }
    std::vector<std::int32_t> counts(codes.size(), 0);
    for (auto i = begin; i < end; ++i) {
      const auto local =
          tabled
              ? table[events.codes[i]]
              : static_cast<std::int32_t>(
                    std::lower_bound(codes.begin(), codes.end(), events.codes[i]) - codes.begin());
      local_codes_[i] = local;
      ++counts[local];
    }
    if (tabled) {
      for (const auto code : codes) table[code] = -1;
    }
    auto& base = bases[s];
    base.resize(codes.size());
    std::int32_t next = 0;
    for (std::size_t k = 0; k < codes.size(); ++k) {
      base[k] = next;
      next += counts[k];
    }
  });
  vocab_begin_.assign(subject_count + 1, 0);
  for (std::size_t s = 0; s < subject_count; ++s) {
    vocab_begin_[s + 1] = vocab_begin_[s] + static_cast<std::int64_t>(vocabs[s].size());
  }
  vocab_.reserve(vocab_begin_.back());
  code_base_.reserve(vocab_begin_.back());
  for (std::size_t s = 0; s < subject_count; ++s) {
    vocab_.insert(vocab_.end(), vocabs[s].begin(), vocabs[s].end());
    code_base_.insert(code_base_.end(), bases[s].begin(), bases[s].end());
  }
}

std::vector<Shard::Job> Shard::Jobs(Rows rows) const {
  std::vector<Job> jobs;
  for (std::int64_t row = 0; row < rows.size;) {
    const auto subject = rows.subjects[row];
    auto end = row + 1;
    while (end < rows.size && rows.subjects[end] == subject) ++end;
    const auto found = std::lower_bound(subjects_.begin(), subjects_.end(), subject);
    const bool known = found != subjects_.end() && *found == subject;
    jobs.push_back({row, end, known ? found - subjects_.begin() : -1});
    row = end;
  }
  return jobs;
}

Result Shard::Aggregate(Rows rows, const std::vector<std::int64_t>& windows, std::uint32_t mask,
                        const std::int32_t* value_columns, std::int64_t value_column_count,
                        int threads) const {
  ValidateRows(rows);
  ValidateWindows(windows);
  ValidateThreads(threads);
  if (!mask || mask >= (1u << kStatisticCount)) {
    throw std::invalid_argument("Invalid statistic mask.");
  }
  const bool counts = mask & (1u << kCodeCount);
  const NumericPlan numeric(mask);
  const auto stats = numeric.stats.size();
  const bool has_numeric = stats > 0;
  if (has_numeric && events_.size && !events_.values) {
    throw std::invalid_argument("Numeric statistics require event values.");
  }
  if (value_columns && value_column_count < 0) {
    throw std::invalid_argument("value_column_count must be nonnegative.");
  }
  // Each row's value columns come out in code order; they need sorting unless
  // the mapping is increasing over the codes it maps.
  bool sort_values = false;
  if (value_columns) {
    std::int32_t previous = -1;
    for (std::int64_t code = 0; code < code_count_; ++code) {
      const auto column = value_columns[code];
      if (column < -1 || column >= value_column_count) {
        throw std::invalid_argument("Value column mapping is outside its block.");
      }
      if (column >= 0) {
        sort_values = sort_values || column <= previous;
        previous = column;
      }
    }
  }
  const auto jobs = Jobs(rows);
  const auto w_count = windows.size();
  const auto stride = static_cast<std::size_t>(rows.size + 1);
  std::vector<Scratch> scratch(WorkerCount(threads, jobs.size()));

  // 1. Size every row: entries at [w * stride + row + 1], prefix-summed below.
  std::vector<std::int64_t> code_outer(counts ? w_count * stride : 0);
  std::vector<std::int64_t> value_outer(has_numeric ? w_count * stride : 0);
  ParallelFor(jobs.size(), threads, [&](std::size_t worker, std::size_t j) {
    const auto& job = jobs[j];
    if (job.subject < 0) return;
    const auto vocab_size = vocab_begin_[job.subject + 1] - vocab_begin_[job.subject];
    SizeRows(events_, local_codes_.data(), event_begin_[job.subject], event_begin_[job.subject + 1],
             vocab_size, rows, job.row_begin, job.row_end, windows, has_numeric, scratch[worker],
             [&](std::size_t w, std::int64_t row, std::int64_t code, std::int64_t value) {
               if (counts) code_outer[w * stride + row + 1] = code;
               if (has_numeric) value_outer[w * stride + row + 1] = value;
             });
  });

  // 2. Allocate each final matrix once. The numeric statistics of a window share
  //    one Csr: it holds the row offsets, the columns and the first statistic's
  //    values, and each further statistic gets only a values array.
  Result result;
  const auto allocate = [&](std::vector<std::int64_t>& outer, std::size_t w,
                            std::int64_t columns) -> Csr& {
    auto* offsets = outer.data() + w * stride;
    for (std::size_t row = 0; row + 1 < stride; ++row) {
      if (offsets[row + 1] > std::numeric_limits<std::int64_t>::max() - offsets[row]) {
        throw std::overflow_error("Output entries exceed int64.");
      }
      offsets[row + 1] += offsets[row];
    }
    auto& csr = result.csrs.emplace_back(rows.size, columns);
    csr.Resize(offsets[stride - 1]);
    std::copy(offsets, offsets + stride, csr.outer());
    result.outputs.push_back({result.csrs.size() - 1, csr.values()});
    return csr;
  };
  std::vector<std::int64_t*> code_inner(w_count), value_inner(w_count);
  std::vector<double*> code_values(w_count), stat_values(w_count * stats);
  for (std::size_t w = 0; w < w_count; ++w) {
    if (counts) {
      const Csr& csr = allocate(code_outer, w, code_count_);
      code_inner[w] = csr.inner();
      code_values[w] = csr.values();
    }
    if (has_numeric) {
      const Csr& csr = allocate(value_outer, w, value_columns ? value_column_count : code_count_);
      const auto index = result.csrs.size() - 1;
      value_inner[w] = csr.inner();
      stat_values[w * stats] = csr.values();
      for (std::size_t k = 1; k < stats; ++k) {
        auto* values = result.shared_values.emplace_back(new double[csr.nnz()]).get();
        stat_values[w * stats + k] = values;
        result.outputs.push_back({index, values});
      }
    }
  }

  // 3. Fill disjoint (subject rows, window) spans in place, largest estimated
  //    work first. A subject's windows are independent tasks, so one dominant
  //    subject spreads over up to one worker per window.
  struct Task {
    std::size_t job, window;
    std::int64_t work;
  };
  std::vector<Task> tasks;
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    if (jobs[j].subject < 0) continue;  // Rows of subjects without events stay empty.
    const auto events = event_begin_[jobs[j].subject + 1] - event_begin_[jobs[j].subject];
    for (std::size_t w = 0; w < w_count; ++w) {
      const auto entries = [&](const std::vector<std::int64_t>& outer) {
        return outer[w * stride + jobs[j].row_end] - outer[w * stride + jobs[j].row_begin];
      };
      std::int64_t work = events;
      if (counts) work += entries(code_outer);
      if (has_numeric) work += static_cast<std::int64_t>(stats) * entries(value_outer);
      tasks.push_back({j, w, work});
    }
  }
  std::stable_sort(tasks.begin(), tasks.end(),
                   [](const Task& a, const Task& b) { return a.work > b.work; });
  scratch.resize(std::max(scratch.size(), WorkerCount(threads, tasks.size())));
  ParallelFor(tasks.size(), threads, [&](std::size_t worker, std::size_t t) {
    const auto& job = jobs[tasks[t].job];
    const auto w = tasks[t].window;
    auto& s = scratch[worker];
    const auto begin = event_begin_[job.subject], end = event_begin_[job.subject + 1];
    const auto* vocab = vocab_.data() + vocab_begin_[job.subject];
    const auto* base = code_base_.data() + vocab_begin_[job.subject];
    const auto vocab_size =
        static_cast<std::size_t>(vocab_begin_[job.subject + 1] - vocab_begin_[job.subject]);
    const auto words = (vocab_size + 63) / 64;
    const double* values = events_.values;
    s.records.assign(vocab_size, 0);
    s.active.assign(words, 0);
    if (has_numeric) {
      s.numbers.assign(vocab_size, 0);
      s.valued.assign(words, 0);
      // Each event is pushed at most once, so a code's slots never outgrow its slice.
      if (numeric.sum || numeric.squares) {
        s.sum_head.assign(base, base + vocab_size);
        s.sum_mid.assign(base, base + vocab_size);
        s.sum_tail.assign(base, base + vocab_size);
        s.slots.resize(end - begin);
        s.suffix_sum.resize(numeric.sum ? end - begin : 0);
        s.suffix_squares.resize(numeric.squares ? end - begin : 0);
        s.back_sum.assign(numeric.sum ? vocab_size : 0, Sum{});
        s.back_squares.assign(numeric.squares ? vocab_size : 0, Sum{});
      }
      if (numeric.min) {
        s.min_head.assign(base, base + vocab_size);
        s.min_tail.assign(base, base + vocab_size);
        s.min_queue.resize(end - begin);
      }
      if (numeric.max) {
        s.max_head.assign(base, base + vocab_size);
        s.max_tail.assign(base, base + vocab_size);
        s.max_queue.resize(end - begin);
      }
    }
    const auto add = [&](std::int64_t i) {
      const auto code = local_codes_[i];
      if (s.records[code]++ == 0) s.active[code >> 6] |= Bit(code);
      if (!has_numeric) return;
      const double value = values[i];
      if (std::isnan(value)) return;
      if (s.numbers[code]++ == 0) s.valued[code >> 6] |= Bit(code);
      if (numeric.sum || numeric.squares) {
        s.slots[s.sum_tail[code]++] = value;
        if (numeric.sum) s.back_sum[code].Add(value);
        if (numeric.squares) s.back_squares[code].Add(value * value);
      }
      const auto offset = static_cast<std::int32_t>(i - begin);
      if (numeric.min) {
        auto& tail = s.min_tail[code];
        while (tail > s.min_head[code] && values[begin + s.min_queue[tail - 1]] >= value) --tail;
        s.min_queue[tail++] = offset;
      }
      if (numeric.max) {
        auto& tail = s.max_tail[code];
        while (tail > s.max_head[code] && values[begin + s.max_queue[tail - 1]] <= value) --tail;
        s.max_queue[tail++] = offset;
      }
    };
    const auto remove = [&](std::int64_t i) {
      const auto code = local_codes_[i];
      if (--s.records[code] == 0) s.active[code >> 6] &= ~Bit(code);
      if (!has_numeric) return;
      const double value = values[i];
      if (std::isnan(value)) return;
      if (--s.numbers[code] == 0) s.valued[code >> 6] &= ~Bit(code);
      if ((numeric.sum || numeric.squares) && s.sum_head[code]++ == s.sum_mid[code]) {
        // The front was empty: the removed value is the back's oldest. Make the
        // back the front, summing its suffixes, then drop the removed slot.
        Sum sum, squares;
        for (auto k = s.sum_tail[code] - 1; k >= s.sum_mid[code]; --k) {
          if (numeric.sum) {
            sum.Add(s.slots[k]);
            s.suffix_sum[k] = sum;
          }
          if (numeric.squares) {
            squares.Add(s.slots[k] * s.slots[k]);
            s.suffix_squares[k] = squares;
          }
        }
        s.sum_mid[code] = s.sum_tail[code];
        if (numeric.sum) s.back_sum[code] = Sum{};
        if (numeric.squares) s.back_squares[code] = Sum{};
      }
      const auto offset = static_cast<std::int32_t>(i - begin);
      if (numeric.min && s.min_head[code] < s.min_tail[code] &&
          s.min_queue[s.min_head[code]] == offset) {
        ++s.min_head[code];
      }
      if (numeric.max && s.max_head[code] < s.max_tail[code] &&
          s.max_queue[s.max_head[code]] == offset) {
        ++s.max_head[code];
      }
    };
    // A code's window sum: its front's suffix from head, plus its back.
    const auto window_sum = [&](const std::vector<Sum>& suffix, const std::vector<Sum>& back,
                                std::int32_t code) {
      const auto head = s.sum_head[code];
      return (head < s.sum_mid[code] ? suffix[head].Value() : 0.0) + back[code].Value();
    };
    const auto statistic = [&](int stat, std::int32_t code) {
      switch (stat) {
        case kValueCount:
          return static_cast<double>(s.numbers[code]);
        case kSum:
          return window_sum(s.suffix_sum, s.back_sum, code);
        case kSumSquared:
          return window_sum(s.suffix_squares, s.back_squares, code);
        case kMin:
          return values[begin + s.min_queue[s.min_head[code]]];
        default:
          return values[begin + s.max_queue[s.max_head[code]]];
      }
    };
    auto right = begin, left = begin;
    for (auto row = job.row_begin; row < job.row_end; ++row) {
      const auto time = rows.times[row];
      for (; right < end && events_.times[right] <= time; ++right) add(right);
      if (HasCutoff(time, windows[w])) {
        const auto cutoff = time - windows[w];
        for (; left < right && events_.times[left] <= cutoff; ++left) remove(left);
      }
      // Walking the set bits in order emits columns in ascending order.
      if (counts) {
        auto position = code_outer[w * stride + row];
        for (std::size_t word = 0; word < words; ++word) {
          for (auto bits = s.active[word]; bits; bits &= bits - 1) {
            const auto code = static_cast<std::int32_t>(word * 64 + __builtin_ctzll(bits));
            code_inner[w][position] = vocab[code];
            code_values[w][position++] = static_cast<double>(s.records[code]);
          }
        }
      }
      if (!has_numeric) continue;
      const auto row_start = value_outer[w * stride + row];
      auto position = row_start;
      for (std::size_t word = 0; word < words; ++word) {
        for (auto bits = s.valued[word]; bits; bits &= bits - 1) {
          const auto code = static_cast<std::int32_t>(word * 64 + __builtin_ctzll(bits));
          std::int64_t column = vocab[code];
          if (value_columns) {
            column = value_columns[column];
            if (column < 0) {
              throw std::invalid_argument("A code with numeric values has no value column.");
            }
          }
          value_inner[w][position] = column;
          for (std::size_t k = 0; k < stats; ++k) {
            const double value = statistic(numeric.stats[k], code);
            if (!std::isfinite(value)) {
              throw std::overflow_error("A numeric sum or sum of squares is not finite.");
            }
            stat_values[w * stats + k][position] = value;
          }
          ++position;
        }
      }
      if (sort_values && position - row_start > 1) {
        // Order the row by value column, permuting every statistic alike.
        s.row_order.clear();
        for (auto p = row_start; p < position; ++p) s.row_order.emplace_back(value_inner[w][p], p);
        std::sort(s.row_order.begin(), s.row_order.end());
        for (std::size_t k = 0; k < stats; ++k) {
          auto* target = stat_values[w * stats + k];
          s.row_values.clear();
          for (const auto& entry : s.row_order) s.row_values.push_back(target[entry.second]);
          std::copy(s.row_values.begin(), s.row_values.end(), target + row_start);
        }
        for (std::size_t p = 0; p < s.row_order.size(); ++p) {
          value_inner[w][row_start + p] = s.row_order[p].first;
        }
      }
    }
  });
  return result;
}

}  // namespace meds_tab
