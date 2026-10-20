// MEDS-Tab's block files: an uncompressed NPZ (a zip archive) holding 'array.npy',
// one homogeneous 3 x nnz array of values, row IDs and column IDs in CSR order,
// and 'shape.npy'. WriteBlock produces exactly the bytes that
// storage.write_csr_block(..., compress=False) produces.
#ifndef MEDS_TAB_NPZ_H_
#define MEDS_TAB_NPZ_H_

#include <cstdint>
#include <string>

namespace meds_tab::npz {

// A borrowed CSR block: outer has rows + 1 offsets from 0 to nnz; inner holds
// each entry's column, ascending within a row. outer and inner share a width.
struct Block {
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t nnz = 0;
  bool index_64 = true;         // outer and inner are int64, else int32.
  bool values_integer = false;  // values are int64, else double.
  const void* outer = nullptr;
  const void* inner = nullptr;
  const void* values = nullptr;
};

// Writes block to path, storing values as integers when `integer` (count
// statistics) and as floats otherwise. Creates missing parent directories.
// Returns the file size. Invalid blocks throw std::invalid_argument; I/O failures
// throw std::system_error.
std::int64_t WriteBlock(const std::string& path, const Block& block, bool integer);

// Writes the selected rows of a block written by WriteBlock to target, as
// WriteBlock does; a negative row selects an empty row. outer holds the source's
// source_rows + 1 row offsets as written. Reads only the selected entries. Throws
// std::runtime_error when the source does not match outer or is malformed.
std::int64_t ExtractRows(const std::string& source, const std::int64_t* outer,
                         std::int64_t source_rows, std::int64_t columns, const std::int64_t* rows,
                         std::int64_t row_count, bool integer, const std::string& target,
                         std::int64_t* written_outer);

}  // namespace meds_tab::npz
#endif
