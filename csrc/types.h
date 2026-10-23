#ifndef MEDS_TAB_TYPES_H_
#define MEDS_TAB_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace meds_tab {

// Timed events sorted by (subject, time). Times are integer microseconds; values
// are NaN when missing and may be null when only code/count is requested.
struct Events {
  const std::int64_t* subjects;
  const std::int64_t* times;
  const std::int32_t* codes;
  const double* values;
  std::int64_t size;
};

// Output rows (subject, anchor time), sorted by (subject, time).
struct Rows {
  const std::int64_t* subjects;
  const std::int64_t* times;
  std::int64_t size;
};

// Statistics in output order. kCodeCount counts records; the others summarize
// non-missing values.
enum Statistic { kCodeCount, kValueCount, kSum, kSumSquared, kMin, kMax, kStatisticCount };

// A CSR matrix that owns its buffers: rows + 1 row offsets (zeroed until filled),
// then nnz columns (ascending within each row) and values. Resize leaves columns
// and values uninitialized, since the engine writes every entry in place.
class Csr {
 public:
  Csr(std::int64_t rows, std::int64_t columns)
      : rows_(rows), columns_(columns), outer_(new std::int64_t[rows + 1]()) {}

  void Resize(std::int64_t nnz) {
    nnz_ = nnz;
    inner_.reset(new std::int64_t[nnz]);
    values_.reset(new double[nnz]);
  }

  std::int64_t rows() const { return rows_; }
  std::int64_t columns() const { return columns_; }
  std::int64_t nnz() const { return nnz_; }
  std::int64_t* outer() const { return outer_.get(); }
  std::int64_t* inner() const { return inner_.get(); }
  double* values() const { return values_.get(); }

 private:
  std::int64_t rows_, columns_, nnz_ = 0;
  std::unique_ptr<std::int64_t[]> outer_, inner_;
  std::unique_ptr<double[]> values_;
};

// The matrices of one engine call. Matrices with the same sparsity pattern (the
// numeric statistics of one window) share one Csr's row offsets and columns:
// each output pairs a Csr with a values array, either the Csr's own or one of
// shared_values.
struct Result {
  struct Output {
    std::size_t csr;
    const double* values;
  };
  std::vector<Csr> csrs;
  std::vector<std::unique_ptr<double[]>> shared_values;
  std::vector<Output> outputs;
};

}  // namespace meds_tab
#endif
