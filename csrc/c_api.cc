// The C ABI declared in meds_tab.h: argument checks, and every exception turned
// into an error value and meds_tab_last_error().
#include "meds_tab.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "npz.h"
#include "shard.h"
#include "static_features.h"
#include "types.h"

struct meds_tab_shard {
  meds_tab::Shard shard;
};
struct meds_tab_result {
  meds_tab::Result result;
};

static_assert(MEDS_TAB_CODE_COUNT == 1u << meds_tab::kCodeCount &&
                  MEDS_TAB_VALUE_COUNT == 1u << meds_tab::kValueCount &&
                  MEDS_TAB_VALUE_SUM == 1u << meds_tab::kSum &&
                  MEDS_TAB_VALUE_SUM_SQD == 1u << meds_tab::kSumSquared &&
                  MEDS_TAB_VALUE_MIN == 1u << meds_tab::kMin &&
                  MEDS_TAB_VALUE_MAX == 1u << meds_tab::kMax,
              "Statistic bits must follow the Statistic enum.");

namespace {
thread_local std::string last_error;

// Runs body and returns its value; on an exception, records its message and
// returns `failure` (`io_failure` for std::system_error).
template <typename Body, typename Failure>
auto Guard(Body body, Failure failure, Failure io_failure) noexcept -> decltype(body()) {
  last_error.clear();
  try {
    return body();
  } catch (const std::system_error& e) {
    try {
      last_error = e.what();
    } catch (...) {
    }
    return io_failure;
  } catch (const std::exception& e) {
    try {
      last_error = e.what();
    } catch (...) {
    }
  } catch (...) {
    last_error = "Unknown C++ exception.";
  }
  return failure;
}

template <typename Body, typename Failure>
auto Guard(Body body, Failure failure) noexcept -> decltype(body()) {
  return Guard(body, failure, failure);
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::vector<std::int64_t> Windows(const std::int64_t* windows, std::int64_t count) {
  Require(count > 0 && windows, "No windows supplied.");
  return {windows, windows + count};
}

meds_tab::Rows ToRows(const meds_tab_rows* rows) {
  Require(rows, "Rows are required.");
  return {rows->subjects, rows->times, rows->size};
}
}  // namespace

extern "C" {
int meds_tab_abi_version(void) { return MEDS_TAB_ABI_VERSION; }
const char* meds_tab_last_error(void) { return last_error.c_str(); }

int meds_tab_rows_sorted(const meds_tab_rows* rows) {
  return Guard(
      [&] {
        meds_tab::ValidateRows(ToRows(rows));
        return 1;
      },
      0);
}

meds_tab_shard* meds_tab_shard_new(const meds_tab_events* events, int64_t code_count,
                                   int32_t threads) {
  return Guard(
      [&] {
        Require(events, "Events are required.");
        const meds_tab::Events view{events->subjects, events->times, events->codes, events->values,
                                    events->size};
        return new meds_tab_shard{meds_tab::Shard(view, code_count, threads)};
      },
      static_cast<meds_tab_shard*>(nullptr));
}

void meds_tab_shard_free(meds_tab_shard* shard) { delete shard; }

meds_tab_result* meds_tab_shard_aggregate(const meds_tab_shard* shard, const meds_tab_rows* rows,
                                          const int64_t* windows, int64_t window_count,
                                          uint32_t statistics, const int32_t* value_columns,
                                          int64_t value_column_count, int32_t threads) {
  return Guard(
      [&] {
        Require(shard, "Invalid shard handle.");
        return new meds_tab_result{
            shard->shard.Aggregate(ToRows(rows), Windows(windows, window_count), statistics,
                                   value_columns, value_column_count, threads)};
      },
      static_cast<meds_tab_result*>(nullptr));
}

meds_tab_result* meds_tab_static(const meds_tab_static_records* records, const meds_tab_rows* rows,
                                 int64_t present_columns, int64_t first_columns) {
  return Guard(
      [&] {
        Require(records, "Static records are required.");
        return new meds_tab_result{meds_tab::AggregateStatic(
            {records->subjects, records->present, records->first, records->values, records->size},
            ToRows(rows), present_columns, first_columns)};
      },
      static_cast<meds_tab_result*>(nullptr));
}

int64_t meds_tab_result_size(const meds_tab_result* result) {
  return result ? static_cast<int64_t>(result->result.outputs.size()) : 0;
}

int meds_tab_result_matrix(const meds_tab_result* result, int64_t index, meds_tab_csr* out) {
  return Guard(
      [&] {
        Require(result && out && index >= 0 && index < meds_tab_result_size(result),
                "Invalid result handle or matrix index.");
        const auto& output = result->result.outputs[index];
        const auto& csr = result->result.csrs[output.csr];
        *out = {csr.rows(), csr.columns(), csr.nnz(),   1,
                0,          csr.outer(),   csr.inner(), output.values};
        return 1;
      },
      0);
}

void meds_tab_result_free(meds_tab_result* result) { delete result; }

int meds_tab_write_block(const char* path, const meds_tab_csr* block, int32_t integer,
                         int64_t* file_bytes) {
  return Guard(
      [&] {
        Require(path && block && file_bytes, "Invalid block arguments.");
        meds_tab::npz::Block view;
        view.rows = block->rows;
        view.columns = block->columns;
        view.nnz = block->nnz;
        view.index_64 = block->index_64 != 0;
        view.values_integer = block->values_integer != 0;
        view.outer = block->outer;
        view.inner = block->inner;
        view.values = block->values;
        *file_bytes = meds_tab::npz::WriteBlock(path, view, integer != 0);
        return 1;
      },
      0, -1);
}

int meds_tab_extract_rows(const char* source, const int64_t* source_outer, int64_t source_rows,
                          int64_t columns, const int64_t* rows, int64_t row_count, int32_t integer,
                          const char* target, int64_t* written_outer, int64_t* file_bytes) {
  return Guard(
      [&] {
        Require(source && source_outer && target && written_outer && file_bytes,
                "Invalid extraction arguments.");
        *file_bytes = meds_tab::npz::ExtractRows(source, source_outer, source_rows, columns, rows,
                                                 row_count, integer != 0, target, written_outer);
        return 1;
      },
      0, -1);
}
}
