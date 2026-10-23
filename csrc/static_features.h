#ifndef MEDS_TAB_STATIC_FEATURES_H_
#define MEDS_TAB_STATIC_FEATURES_H_

#include <cstdint>

#include "types.h"

namespace meds_tab {

// Kept timeless records in file order (subjects need not be grouped).
struct StaticRecords {
  const std::int64_t* subjects;
  const std::int32_t* present;  // static/present column of the record.
  const std::int32_t* first;    // static/first column, or -1 when not a selected feature.
  const double* values;         // NaN = missing; may be null when every first is -1.
  std::int64_t size;
};

// Two matrices over rows (sorted by subject): static/present (1 per code with a
// record) and static/first (first non-missing value in file order). Each row
// receives its subject's entries; subjects without records get empty rows.
Result AggregateStatic(StaticRecords records, Rows rows, std::int64_t present_columns,
                       std::int64_t first_columns);

}  // namespace meds_tab
#endif
