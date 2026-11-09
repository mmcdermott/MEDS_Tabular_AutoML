#ifndef MEDS_TAB_SHARD_H_
#define MEDS_TAB_SHARD_H_

#include <cstdint>
#include <vector>

#include "types.h"

namespace meds_tab {

// One shard's timed events, validated and indexed once, then aggregated for any
// rows and windows. The event arrays are borrowed and must outlive the shard.
//
// The index renumbers each subject's codes densely in sorted order and records,
// for every event, its subject-local code. It costs 4 bytes per event, 8 per
// distinct (subject, code) and 24 per subject.
class Shard {
 public:
  // Validates the events and indexes subjects on up to `threads` threads.
  Shard(Events events, std::int64_t code_count, int threads);

  // Matrices window-major, then the statistics selected by `mask` in Statistic
  // order. Window widths are positive microseconds, or -1 for all history; a
  // finite window at time t covers (t - width, t]. value_columns (code_count
  // entries, or null for the identity) maps each code to its value-block column,
  // or to -1 for codes whose values are not features; code/count columns are
  // code IDs. Rows of subjects without events are empty.
  Result Aggregate(Rows rows, const std::vector<std::int64_t>& windows, std::uint32_t mask,
                   const std::int32_t* value_columns, std::int64_t value_column_count,
                   int threads) const;

 private:
  // The rows of one subject, and the subject's index in subjects_ (-1 when the
  // subject has no events).
  struct Job {
    std::int64_t row_begin, row_end;
    std::int64_t subject;
  };
  std::vector<Job> Jobs(Rows rows) const;

  Events events_;
  std::int64_t code_count_;
  // Subject s has events [event_begin_[s], event_begin_[s + 1]) and codes
  // vocab_[vocab_begin_[s]] to vocab_[vocab_begin_[s + 1] - 1].
  std::vector<std::int64_t> subjects_;  // Distinct subjects, ascending.
  std::vector<std::int64_t> event_begin_;
  std::vector<std::int64_t> vocab_begin_;
  std::vector<std::int32_t> vocab_;  // Each subject's distinct codes, ascending.
  // Aligned with vocab_: where the code's events start when the subject's events
  // are grouped by code. Each code's min/max queue lives in that slice.
  std::vector<std::int32_t> code_base_;
  std::vector<std::int32_t> local_codes_;  // Per event: its code's index in its subject's vocab.
};

// Throws unless rows are well formed and sorted by (subject, time).
void ValidateRows(Rows rows);

}  // namespace meds_tab
#endif
