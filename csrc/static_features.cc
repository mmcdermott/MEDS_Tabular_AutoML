#include "static_features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "shard.h"

namespace meds_tab {

Result AggregateStatic(StaticRecords records, Rows rows, std::int64_t present_columns,
                       std::int64_t first_columns) {
  ValidateRows(rows);
  if (records.size < 0 || present_columns < 0 || first_columns < 0) {
    throw std::invalid_argument("Static sizes must be nonnegative.");
  }
  if (records.size && (!records.subjects || !records.present || !records.first)) {
    throw std::invalid_argument("Nonempty static records require subject/column arrays.");
  }
  for (std::int64_t i = 0; i < records.size; ++i) {
    if (records.present[i] < 0 || records.present[i] >= present_columns || records.first[i] < -1 ||
        records.first[i] >= first_columns) {
      throw std::invalid_argument("Static record column is outside its block.");
    }
    if (records.first[i] >= 0) {
      if (!records.values) throw std::invalid_argument("static/first requires values.");
      if (std::isinf(records.values[i])) {
        throw std::invalid_argument("Infinite values are unsupported.");
      }
    }
  }
  // Group records by subject, keeping file order inside each subject.
  std::vector<std::int64_t> order(records.size);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
    return records.subjects[a] < records.subjects[b];
  });
  std::vector<std::int64_t> subjects;
  std::vector<std::int64_t> present_begin{0}, first_begin{0};
  std::vector<std::int32_t> present_cols, first_cols;
  std::vector<double> first_values;
  std::vector<std::pair<std::int32_t, double>> firsts;
  for (std::int64_t i = 0; i < records.size;) {
    const auto subject = records.subjects[order[i]];
    auto end = i;
    const auto present_start = present_cols.size();
    firsts.clear();
    for (; end < records.size && records.subjects[order[end]] == subject; ++end) {
      const auto r = order[end];
      present_cols.push_back(records.present[r]);
      const auto column = records.first[r];
      if (column >= 0 && !std::isnan(records.values[r]) &&
          std::none_of(firsts.begin(), firsts.end(),
                       [&](const auto& entry) { return entry.first == column; })) {
        firsts.emplace_back(column, records.values[r]);
      }
    }
    std::sort(present_cols.begin() + present_start, present_cols.end());
    present_cols.erase(std::unique(present_cols.begin() + present_start, present_cols.end()),
                       present_cols.end());
    std::sort(firsts.begin(), firsts.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [column, value] : firsts) {
      first_cols.push_back(column);
      first_values.push_back(value);
    }
    subjects.push_back(subject);
    present_begin.push_back(static_cast<std::int64_t>(present_cols.size()));
    first_begin.push_back(static_cast<std::int64_t>(first_cols.size()));
    i = end;
  }
  // Rows sorted by subject: advance one subject cursor.
  std::vector<std::int64_t> row_subject(rows.size, -1);
  std::size_t cursor = 0;
  for (std::int64_t row = 0; row < rows.size; ++row) {
    while (cursor < subjects.size() && subjects[cursor] < rows.subjects[row]) ++cursor;
    if (cursor < subjects.size() && subjects[cursor] == rows.subjects[row]) {
      row_subject[row] = static_cast<std::int64_t>(cursor);
    }
  }
  Result result;
  const auto build = [&](std::int64_t columns, const std::vector<std::int64_t>& begin,
                         const std::vector<std::int32_t>& cols, const double* values) {
    auto& matrix = result.csrs.emplace_back(rows.size, columns);
    std::int64_t total = 0;
    for (auto s : row_subject) {
      if (s < 0) continue;
      const auto size = begin[s + 1] - begin[s];
      if (size > std::numeric_limits<std::int64_t>::max() - total) {
        throw std::overflow_error("Output entries exceed int64.");
      }
      total += size;
    }
    matrix.Resize(total);
    auto* outer = matrix.outer();
    outer[0] = 0;
    for (std::int64_t row = 0; row < rows.size; ++row) {
      const auto s = row_subject[row];
      const auto size = s < 0 ? 0 : begin[s + 1] - begin[s];
      if (size) {
        std::copy(cols.begin() + begin[s], cols.begin() + begin[s + 1],
                  matrix.inner() + outer[row]);
        for (std::int64_t k = 0; k < size; ++k) {
          matrix.values()[outer[row] + k] = values ? values[begin[s] + k] : 1.0;
        }
      }
      outer[row + 1] = outer[row] + size;
    }
    result.outputs.push_back({result.csrs.size() - 1, matrix.values()});
  };
  build(present_columns, present_begin, present_cols, nullptr);
  build(first_columns, first_begin, first_cols, first_values.data());
  return result;
}

}  // namespace meds_tab
