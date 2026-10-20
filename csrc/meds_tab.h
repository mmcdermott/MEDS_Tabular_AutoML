/* The C ABI of MEDS-Tab's native engine, loaded with ctypes by
 * src/MEDS_tabular_automl/native/engine.py.
 *
 * Errors: functions returning a pointer return NULL on failure, and functions
 * returning int return 0 (or -1 for I/O failures); meds_tab_last_error() then
 * describes the failure. The message is thread-local and valid until the next
 * call on the same thread. No C++ exception crosses this boundary.
 *
 * Ownership: every input array is borrowed. A shard borrows its event arrays for
 * its whole lifetime; all other inputs are borrowed only during the call.
 * Returned handles belong to the caller and are released with the matching
 * *_free function, which accepts NULL. A result's matrix buffers remain valid
 * until meds_tab_result_free.
 *
 * Arrays: a pointer may be NULL only when its length is zero. Times are integer
 * microseconds. Window widths are positive, or -1 for all history. Shards and
 * results are never mutated after creation, so they may be read concurrently.
 */
#ifndef MEDS_TAB_H_
#define MEDS_TAB_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEDS_TAB_ABI_VERSION 1

typedef struct meds_tab_shard meds_tab_shard;
typedef struct meds_tab_result meds_tab_result;

/* Timed events sorted by (subject, time); codes lie in [0, code_count). values:
 * NaN is missing; NULL is allowed when only code/count is requested. */
typedef struct {
  const int64_t* subjects;
  const int64_t* times;
  const int32_t* codes;
  const double* values;
  int64_t size;
} meds_tab_events;

/* Output rows (subject, time), sorted by (subject, time). */
typedef struct {
  const int64_t* subjects;
  const int64_t* times;
  int64_t size;
} meds_tab_rows;

/* Selected timeless records in file order. present is the record's static/present
 * column; first is its static/first column, or -1 when the code is not a selected
 * static/first feature. values (NaN is missing) may be NULL when every first is -1. */
typedef struct {
  const int64_t* subjects;
  const int32_t* present;
  const int32_t* first;
  const double* values;
  int64_t size;
} meds_tab_static_records;

/* A CSR matrix: outer has rows + 1 offsets; inner (ascending within each row) and
 * values have nnz entries. index_64 selects int64 outer and inner (else int32);
 * values_integer selects int64 values (else double). */
typedef struct {
  int64_t rows;
  int64_t columns;
  int64_t nnz;
  int32_t index_64;
  int32_t values_integer;
  const void* outer;
  const void* inner;
  const void* values;
} meds_tab_csr;

int meds_tab_abi_version(void);
const char* meds_tab_last_error(void);

/* 1 if rows are sorted by (subject, time), else 0. */
int meds_tab_rows_sorted(const meds_tab_rows* rows);

/* Validates and indexes one shard's events on up to `threads` threads. */
meds_tab_shard* meds_tab_shard_new(const meds_tab_events* events, int64_t code_count,
                                   int32_t threads);
void meds_tab_shard_free(meds_tab_shard* shard);

/* Statistic bits, in output order. */
#define MEDS_TAB_CODE_COUNT (1u << 0)
#define MEDS_TAB_VALUE_COUNT (1u << 1)
#define MEDS_TAB_VALUE_SUM (1u << 2)
#define MEDS_TAB_VALUE_SUM_SQD (1u << 3)
#define MEDS_TAB_VALUE_MIN (1u << 4)
#define MEDS_TAB_VALUE_MAX (1u << 5)

/* One matrix per window and selected statistic: window-major, then statistics in
 * bit order. A finite window at time t covers (t - width, t]. value_columns
 * (code_count entries, or NULL for the identity) maps each code to its
 * value-block column, or to -1 for codes whose values are not features. */
meds_tab_result* meds_tab_shard_aggregate(const meds_tab_shard* shard, const meds_tab_rows* rows,
                                          const int64_t* windows, int64_t window_count,
                                          uint32_t statistics, const int32_t* value_columns,
                                          int64_t value_column_count, int32_t threads);

/* Two matrices, static/present then static/first, with each row holding its
 * subject's features: 1 for every present code, and the first non-missing value
 * in file order of every first code. */
meds_tab_result* meds_tab_static(const meds_tab_static_records* records, const meds_tab_rows* rows,
                                 int64_t present_columns, int64_t first_columns);

int64_t meds_tab_result_size(const meds_tab_result* result);
int meds_tab_result_matrix(const meds_tab_result* result, int64_t index, meds_tab_csr* out);
void meds_tab_result_free(meds_tab_result* result);

/* Writes block to path as an uncompressed MEDS-Tab block file, byte for byte as
 * storage.write_csr_block(..., compress=False) writes it. integer stores values
 * as integers (count statistics). *file_bytes receives the file size. */
int meds_tab_write_block(const char* path, const meds_tab_csr* block, int32_t integer,
                         int64_t* file_bytes);

/* Writes the selected rows of a block written by meds_tab_write_block to target,
 * as meds_tab_write_block would, reading only their entries. source_outer holds
 * the source's source_rows + 1 row offsets; a negative row selects an empty row.
 * written_outer (row_count + 1) receives the target's row offsets. */
int meds_tab_extract_rows(const char* source, const int64_t* source_outer, int64_t source_rows,
                          int64_t columns, const int64_t* rows, int64_t row_count, int32_t integer,
                          const char* target, int64_t* written_outer, int64_t* file_bytes);

#ifdef __cplusplus
}
#endif
#endif
