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

#define MEDS_TAB_ABI_VERSION 3

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
 * column; mean is its static/mean column, or -1 when the code is not a selected
 * static/mean feature. values (NaN is missing) may be NULL when every mean is -1. */
typedef struct {
  const int64_t* subjects;
  const int32_t* present;
  const int32_t* mean;
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
/* 1 / (1 + days since the code's last record in the window): code/recency. */
#define MEDS_TAB_CODE_RECENCY (1u << 6)

/* One matrix per window and selected statistic: window-major, then statistics in
 * bit order. A finite window at time t covers (t - width, t]. value_columns
 * (code_count entries, or NULL for the identity) maps each code to its
 * value-block column, or to -1 for codes whose values are not features. */
meds_tab_result* meds_tab_shard_aggregate(const meds_tab_shard* shard, const meds_tab_rows* rows,
                                          const int64_t* windows, int64_t window_count,
                                          uint32_t statistics, const int32_t* value_columns,
                                          int64_t value_column_count, int32_t threads);

/* Two matrices, static/present then static/mean, with each row holding its
 * subject's features: 1 for every present code, and the mean of the non-missing
 * values (summed in file order) of every mean code. */
meds_tab_result* meds_tab_static(const meds_tab_static_records* records, const meds_tab_rows* rows,
                                 int64_t present_columns, int64_t mean_columns);

int64_t meds_tab_result_size(const meds_tab_result* result);
int meds_tab_result_matrix(const meds_tab_result* result, int64_t index, meds_tab_csr* out);
void meds_tab_result_free(meds_tab_result* result);

/* Writes block to path as an uncompressed MEDS-Tab block file, byte for byte as
 * storage.write_csr_block(..., compress=False) writes it. integer stores values
 * as integers (count statistics). *file_bytes receives the file size. */
int meds_tab_write_block(const char* path, const meds_tab_csr* block, int32_t integer,
                         int64_t* file_bytes);

/* Writes count blocks as meds_tab_write_block does, on up to threads threads:
 * blocks[i] to paths[i], integer[i] for its values; file_bytes[i] receives its
 * size. One call writes a shard's blocks without returning to the caller (and,
 * from Python, without taking the GIL) between them. On failure, the blocks
 * written so far stay on disk. */
int meds_tab_write_blocks(int64_t count, const char* const* paths, const meds_tab_csr* blocks,
                          const int32_t* integer, int64_t* file_bytes, int32_t threads);
/* meds-tab-describe's code frequencies, counted from Arrow record batches. */
typedef struct meds_tab_code_counter meds_tab_code_counter;

/* One column of an Arrow array: validity bitmap (NULL: no nulls), data buffer,
 * and the array's offset in elements. */
typedef struct {
  const uint8_t* validity;
  const void* data;
  int64_t offset;
} meds_tab_arrow_column;

/* The columns of one record batch of a MEDS shard. subjects: int64. times: only
 * the validity is read (null: a static record). code_indices: int32 indices into
 * the batch's string dictionary, whose offsets (int32, or int64 when
 * large_offsets) and validity are code_offsets and whose bytes are code_bytes.
 * values: float64 when values_64, else float32; null or NaN is missing. */
typedef struct {
  int64_t length;
  meds_tab_arrow_column subjects;
  meds_tab_arrow_column times;
  meds_tab_arrow_column code_indices;
  meds_tab_arrow_column code_offsets;
  const char* code_bytes;
  int64_t dictionary_length;
  int32_t large_offsets;
  meds_tab_arrow_column values;
  int32_t values_64;
} meds_tab_arrow_records;

/* A counter of the records of the subjects in train_subjects (sorted ascending;
 * copied); NULL on error. */
meds_tab_code_counter* meds_tab_code_counter_new(const int64_t* train_subjects,
                                                 int64_t train_count);
void meds_tab_code_counter_free(meds_tab_code_counter* counter);

/* Adds count batches' records of training subjects with a subject and a code:
 * per code, timed records, timed records with a value, timeless records and
 * timeless records with a value (describe's code, value, static/present and
 * static/mean counts). Codes are interned once per distinct dictionary string;
 * a dictionary repeating the previous batch's is not looked up again. */
int meds_tab_code_counter_add(meds_tab_code_counter* counter, const meds_tab_arrow_records* batches,
                              int64_t count);

/* Sets *found to 1 if any of subjects (size values) is a training subject, else 0.
 * Values in null slots may be anything: a match there only makes *found 1. */
int meds_tab_code_counter_has_train_subjects(const meds_tab_code_counter* counter,
                                             const int64_t* subjects, int64_t size, int32_t* found);

/* Adds from's counts to into's, matching codes by name. */
int meds_tab_code_counter_merge(meds_tab_code_counter* into, const meds_tab_code_counter* from);

/* The number of distinct codes and the total bytes of their names. */
int meds_tab_code_counter_size(const meds_tab_code_counter* counter, int64_t* codes,
                               int64_t* name_bytes);

/* Copies the codes out: name_offsets (codes + 1), names (name_bytes), counts
 * (codes x 4, code-major), sized from meds_tab_code_counter_size. */
int meds_tab_code_counter_export(const meds_tab_code_counter* counter, int64_t* name_offsets,
                                 char* names, int64_t* counts);

#ifdef __cplusplus
}
#endif
#endif
